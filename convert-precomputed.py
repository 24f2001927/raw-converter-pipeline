#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "tensorstore",
#     "numpy",
#     "tqdm",
# ]
# ///

"""
Neuroglancer Precomputed Converter — TensorStore Integrated
================================================================
Strategy:
  - Uses `tensorstore` to automatically handle multiscale metadata,
    XYZ chunk formatting, and async disk writing (from Script 1).
  - Uses `np.memmap` to safely stream the massive ZYX .raw file
    without loading it entirely into RAM (from Script 2).
  - Bypasses the need for manual accumulators and thread pools;
    TensorStore's C++ backend handles the concurrent I/O.
"""

import logging
import os
import time

import numpy as np
import tensorstore as ts
from tqdm import tqdm

# ─── Config ───────────────────────────────────────────────────────────────────

RAW_FILE_PATH = "path/to/raw/file"
OUTPUT_DIR = "neuroglancer_precomputed"
SHAPE_ZYX = (706, 20000, 20000)
CHUNK_XYZ = [512, 512, 512]
RESOLUTION = [4000, 4000, 60000]

NUM_SCALES = 3  # Number of resolution scales to initialize
BLOCK_Y = 512  # Y-strip height
BLOCK_Z = 512  # Aligning Z-block with chunk size avoids TS cache thrashing

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def convert():
    nz, ny, nx = SHAPE_ZYX
    base_shape = np.array([nx, ny, nz, 1])  # Tensorstore uses XYZC
    base_res = np.array(RESOLUTION)

    log.info("=" * 60)
    log.info("TENSORSTORE INTEGRATED PRECOMPUTED CONVERTER")
    log.info("=" * 60)
    log.info(f"  Input      : {RAW_FILE_PATH}")
    log.info(f"  Shape      : {SHAPE_ZYX} (ZYX) → ({nx}, {ny}, {nz}) (XYZ)")
    log.info(f"  Chunks     : {CHUNK_XYZ}")
    log.info(f"  Scales     : {NUM_SCALES}")
    log.info("=" * 60)

    dataset_scale_0 = None

    log.info("Initializing TensorStore multiscale metadata...")
    for scale in range(NUM_SCALES):
        # Downsample X and Y by 2^scale, leave Z and Channel untouched
        downsample_factors = np.array([2**scale, 2**scale, 1, 1])

        scale_shape = np.ceil(base_shape / downsample_factors).astype(int)
        scale_res = base_res * downsample_factors[:-1]

        # Open tensorstore dataset for this scale (creates directories and info file)
        store = ts.open(
            {
                "driver": "neuroglancer_precomputed",
                "kvstore": {"driver": "file", "path": OUTPUT_DIR},
                "multiscale_metadata": {
                    "data_type": "uint8",
                    "num_channels": 1,
                    "type": "image",
                },
                "scale_metadata": {
                    "size": scale_shape[:-1].tolist(),
                    "resolution": scale_res.tolist(),
                    "chunk_size": CHUNK_XYZ,
                    "encoding": "raw",
                },
            },
            create=True,
            dtype=ts.uint8,
            shape=scale_shape.tolist(),
        ).result()

        # Keep a reference to the base scale (scale 0) for data ingestion
        if scale == 0:
            dataset_scale_0 = store

    log.info("✓ Metadata and directories initialized.")

    if not os.path.exists(RAW_FILE_PATH):
        log.warning(
            f"File {RAW_FILE_PATH} not found. Ensure the path is correct before running."
        )
        return

    log.info("Starting high-performance chunked conversion...")
    t_start = time.time()

    # Calculate total chunks for the progress bar
    BLOCK_X = CHUNK_XYZ[0]
    total_chunks = (
        ((nx + BLOCK_X - 1) // BLOCK_X)
        * ((ny + BLOCK_Y - 1) // BLOCK_Y)
        * ((nz + BLOCK_Z - 1) // BLOCK_Z)
    )

    with (
        open(RAW_FILE_PATH, "rb") as f,
        tqdm(total=total_chunks, desc="Writing Chunks", unit="chk") as pbar,
    ):
        # 1. Outer Loop: Z-blocks (To read slices sequentially)
        for z_start in range(0, nz, BLOCK_Z):
            z_end = min(z_start + BLOCK_Z, nz)
            actual_z = z_end - z_start

            # 2. Middle Loop: Y-strips
            for y_start in range(0, ny, BLOCK_Y):
                y_end = min(y_start + BLOCK_Y, ny)
                actual_y = y_end - y_start

                # Allocate a buffer for this Z, Y strip across ALL X (~5.2GB peak RAM)
                buffer_zyx = np.empty((actual_z, actual_y, nx), dtype=np.uint8)

                for z_offset in range(actual_z):
                    z_current = z_start + z_offset
                    # Exact byte offset for this specific Y-row in this Z-slice
                    offset = z_current * ny * nx + y_start * nx
                    f.seek(offset)

                    # Read directly into the pre-allocated numpy memory view (Zero-copy)
                    bytes_to_read = actual_y * nx
                    bytes_read = f.readinto(memoryview(buffer_zyx[z_offset, :, :]))
                    if bytes_read != bytes_to_read:
                        raise EOFError(
                            f"Unexpected end of file at Z={z_current}, Y={y_start}"
                        )

                futures = []

                # 3. Inner Loop: Process X in chunks of 512
                for x_start in range(0, nx, BLOCK_X):
                    x_end = min(x_start + BLOCK_X, nx)

                    # Slice 512x512x512 block out of the RAM buffer
                    chunk_zyx = buffer_zyx[:, :, x_start:x_end]

                    # Transpose only this small 128MB chunk (Extremely CPU-cache friendly!)
                    chunk_xyz = np.ascontiguousarray(np.transpose(chunk_zyx, (2, 1, 0)))
                    chunk_xyz_c = np.expand_dims(chunk_xyz, axis=3)

                    # Dispatch async write to TensorStore
                    future = dataset_scale_0[
                        x_start:x_end, y_start:y_end, z_start:z_end, 0:1
                    ].write(chunk_xyz_c)
                    futures.append(future)

                # Wait for all chunks in this Y-strip to safely write to disk before moving on
                for future in futures:
                    future.result()
                    pbar.update(1)

    elapsed = time.time() - t_start
    log.info("=" * 60)
    log.info("CONVERSION COMPLETE")
    log.info("=" * 60)
    log.info(f"  Total time : {elapsed:.1f}s ({elapsed / 60:.1f}m)")
    log.info(f"  Throughput : {total_chunks / max(elapsed, 0.001):.1f} chunks/s")
    log.info("  Scale 0 base resolution successfully populated.")
    log.info("Done.")


if __name__ == "__main__":
    convert()
