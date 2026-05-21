# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pyvista",
#     "caveclient",
#     "meshparty",
#     "trimesh",
# ]
# ///

import numpy as np
import pyvista as pv
import caveclient
from meshparty import trimesh_io

# ==========================================
# 1. Authenticate and Connect to MICrONS
# ==========================================
# Connect to the public MICrONS mm^3 release (minnie65)
client = caveclient.CAVEclient('minnie65_public')

# Note: The first time you run this, you will need a CAVE token.
# You can get one for free at microns-explorer.org and set it via:
# client.auth.save_token(token="YOUR_TOKEN_HERE")

# ==========================================
# 2. Define the "Perfect Bounding Box"
# ==========================================
# Coordinates are in nanometers. This defines the absolute limits of your cube.
# We are taking a tight spatial crop deep in the volume.
X_MIN, X_MAX = 170000, 200000
Y_MIN, Y_MAX = 170000, 200000
Z_MIN, Z_MAX = 200000, 230000

bounds = [X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX]

# ==========================================
# 3. Query the Connectome
# ==========================================
# In a full script, you would query the 'nucleus_neuron_svm' table for all 
# cells inside your bounding box. For this pipeline, we will use a hardcoded 
# list of known, beautiful pyramidal Root IDs.
root_ids = [
    864691135815589007, 
    864691135939414017,
    864691135501863005 
]

# ==========================================
# 4. Stream Meshes & Convert to PyVista
# ==========================================
# Get the cloud volume endpoint for the meshes
mesh_endpoint = client.info.segmentation_source()
mm = trimesh_io.MeshMeta(cv_path=mesh_endpoint)

plotter = pv.Plotter(off_screen=False)
plotter.set_background('black')

print("Downloading and slicing meshes... this might take a minute.")

for root_id in root_ids:
    # Fetch the raw mesh directly from the Google Cloud bucket
    mesh = mm.mesh(seg_id=root_id)
    
    # Meshparty returns a Trimesh object. We must convert it to a 
    # PyVista PolyData object to perform the boolean cube cut.
    # PyVista requires a specific face array padding: [3, v1, v2, v3, 3, v4...]
    faces = np.column_stack((np.full(len(mesh.faces), 3), mesh.faces)).flatten()
    pv_mesh = pv.PolyData(mesh.vertices, faces)
    
    # ==========================================
    # 5. THE SURGICAL CUT
    # ==========================================
    # This is the magic command. It strictly crops the sprawling dendritic 
    # and axonal arbors exactly at the borders of your defined box.
    clipped_mesh = pv_mesh.clip_box(bounds, invert=False)
    
    # Add to the scene. Emissive colors look best for this type of dataviz.
    plotter.add_mesh(clipped_mesh, color='cyan', opacity=0.8, smooth_shading=True)

# ==========================================
# 6. Render the Final Environment
# ==========================================
# Add the glass bounding box outline to emphasize the "tissue slice" aesthetic
box = pv.Box(bounds=bounds)
plotter.add_mesh(box, style='wireframe', color='white', line_width=2)

# Set a cinematic isometric camera angle and launch
plotter.camera_position = 'iso'
plotter.show()
