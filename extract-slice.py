import os

import numpy as np
import tifffile

# 1. Configuration matching your NRRD/Raw details
raw_file = "path/to/raw/file"
output_dir = "extracted_slices"
shape = (z, y, x)  # use this and put correct shape.
dtype = np.uint8

# Create the output directory if it doesn't already exist
os.makedirs(output_dir, exist_ok=True)

print(f"Memory-mapping the raw file: {raw_file}...")
# 2. Memory-map the data. (mode='r' ensures we only read, protecting the original data)
mapped_data = np.memmap(raw_file, dtype=dtype, mode="r", shape=shape)

total_slices = shape[0]
print(f"Starting extraction of {total_slices} slices. Saving to '{output_dir}/' ...")

# 3. Loop through the Z-axis and save each slice
for z in range(total_slices):
    print(f"Saving slice {z + 1}/{total_slices}...")

    # Extract the 2D slice (this pulls this specific 20000x20000 plane into RAM)
    single_slice = mapped_data[z, :, :]

    # Define output filename with zero-padding (e.g., slice_000.tif to slice_508.tif)
    output_filename = os.path.join(output_dir, f"slice_{z:03d}.tif")

    # Save the slice as a TIFF image
    # Using compression='zlib' saves disk space, but you can remove it if you want faster saving speeds
    tifffile.imwrite(output_filename, single_slice, compression="zlib")

print("All slices extracted successfully!")
