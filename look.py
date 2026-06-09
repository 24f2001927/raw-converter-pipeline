import numpy as np

# z,y,x format in the shape and only works z sliced
shape = (z, y, x)  # make sure your raw file in z stacked or else change that
raw_file = "path/to/raw/volume"

print("Scanning volume for data...")
mapped_data = np.memmap(raw_file, dtype=np.uint8, mode="r", shape=shape)

for z in range(0, 815, 50):
    slice_max = np.max(mapped_data[z, :, :])
    print(f"Z-Slice {z} maximum pixel value: {slice_max}")
