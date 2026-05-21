import cloudvolume
import numpy as np

h01_mesh_path = 'gs://h01-release/data/20210601/c3'
vol = cloudvolume.CloudVolume(h01_mesh_path, use_https=True)

# Fetch a tiny bounding box in the volume to find real segment IDs
print("Fetching bounding box to find real IDs...")
# H01 volume info: x: 1-326400, y: 1-229376, z: 1-5250
# Increase cutout size significantly to get at least 120 neurons
cutout = vol[150000:150500, 150000:150500, 2000:2005]
valid_ids = np.unique(cutout)
valid_ids = valid_ids[valid_ids != 0]

print(f"Found IDs: {list(valid_ids)[:120]}")
