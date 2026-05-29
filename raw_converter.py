import dask.array as da
import numpy as np
import zarr
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image

# 1. Define dimensions from your NRRD header
# NRRD lists as (X, Y, Z). Numpy requires C-contiguous order: (Z, Y, X)
# uncomment below and add the shapeas z, y, x
# shape = (1762, 20000, 20000)
dtype = np.uint8
raw_file = "path/to/file"
output_zarr = "volume.zarr"

print(f"Memory-mapping the raw file: {raw_file}...")
# 2. Memory-map the data. This acts like an array but reads straight from disk, saving your RAM.
mapped_data = np.memmap(raw_file, dtype=dtype, mode="r", shape=shape)

# 3. Wrap in a Dask array for chunked processing
# Neuroglancer performs best with chunks around 64^3 to 128^3.
# We'll use (16, 2000, 2000) here to balance processing speed and read efficiency.
print("Wrapping in Dask array...")
# Just grab the first 10 slices and a 2000x2000 corner of the image
# dask_array = da.from_array(mapped_data[:10, :2000, :2000], chunks=(5, 500, 500))
dask_array = da.from_array(mapped_data, chunks=(16, 2000, 2000))
# Grab 10 slices from the dead center of the Z axis (around slice 400)
# and a 2000x2000 square from the dead center of the X/Y axes (around pixel 10,000)
# z_start, z_end = 400, 410
# y_start, y_end = 9000, 11000
# x_start, x_end = 9000, 11000

# test_slice = mapped_data[z_start:z_end, y_start:y_end, x_start:x_end]

# # Check if there is actual data here!
# # If this prints '0', we are still in empty space.
# print(f"Maximum pixel value in this slice: {np.max(test_slice)}")

# # Wrap the central slice in Dask
# dask_array = da.from_array(test_slice, chunks=(10, 500, 500))
# 4. Prepare the Zarr store
# 4. Prepare the Zarr store
print(f"Creating OME-Zarr store at: {output_zarr}...")
# In Zarr 3, LocalStore replaces DirectoryStore
store = zarr.storage.LocalStore(output_zarr)
root = zarr.group(store=store, overwrite=True)

# 5. Define a scaler to create the multiresolution pyramid (crucial for Neuroglancer)
# max_layer=4 means it will create the original + 4 downsampled levels
# method='nearest' is the fastest for a dataset this massive
scaler = Scaler(max_layer=4, method="nearest")

# 6. Write the image out to disk
print(
    "Writing data to Zarr... (This will take a significant amount of time based on disk speed)"
)
write_image(image=dask_array, group=root, axes="zyx", scaler=scaler)

print("Conversion complete!")
