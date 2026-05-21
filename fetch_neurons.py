# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "cloud-volume",
#     "numpy",
#     "trimesh",
# ]
# ///

import cloudvolume
import numpy as np
import trimesh
import os

def fetch_neurons(num_neurons=15):
    h01_mesh_path = 'gs://h01-release/data/20210601/c3'
    vol = cloudvolume.CloudVolume(h01_mesh_path, use_https=True)
    
    # Fetch a bounding box in the volume to find real segment IDs
    print("Fetching bounding box to find real IDs...")
    cutout = vol[150000:150500, 150000:150500, 2000:2005]
    valid_ids = np.unique(cutout)
    valid_ids = valid_ids[valid_ids != 0]
    
    target_ids = list(valid_ids)[:num_neurons]
    print(f"Targeting {len(target_ids)} neurons...")
    
    os.makedirs("h01_meshes", exist_ok=True)
    
    for i, seg_id in enumerate(target_ids):
        try:
            print(f"Fetching mesh {i+1}/{len(target_ids)} (ID: {seg_id})...")
            meshes = vol.mesh.get(seg_id)
            mesh_data = meshes[seg_id]
            
            # Create Trimesh object
            tmesh = trimesh.Trimesh(vertices=mesh_data.vertices, faces=mesh_data.faces)
            out_path = f"h01_meshes/neuron_{seg_id}.obj"
            tmesh.export(out_path)
            print(f"Saved {out_path}")
        except Exception as e:
            print(f"Error fetching mesh for ID {seg_id}: {e}")

if __name__ == "__main__":
    fetch_neurons(15)
