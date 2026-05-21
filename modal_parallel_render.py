import modal
import io
import numpy as np

# Define a robust image with all necessary 3D rendering and parallel dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("xvfb", "libgl1-mesa-glx", "libglib2.0-0")
    .uv_pip_install("cloud-volume", "numpy", "pyvista[jupyter]", "pillow")
)

app = modal.App("h01-parallel-render")

# The exact spatial coordinates for our bounding box in nanometers
# H01 Resolution: 8x8x33 nm
X_RES, Y_RES, Z_RES = 8, 8, 33
BOUNDS = [150000*X_RES, 150500*X_RES, 150000*Y_RES, 150500*Y_RES, 2000*Z_RES, 2060*Z_RES]

@app.function(image=image, memory=4096)
def get_neuron_ids():
    """Step 1: Efficiently query the dense voxel segmentation to find all unique IDs in the prism."""
    import cloudvolume
    vol = cloudvolume.CloudVolume('gs://h01-release/data/20210601/c3', use_https=True)
    print("Scanning massive 3D volume for unique segments...")
    cutout = vol[150000:150500, 150000:150500, 2000:2060]
    valid_ids = np.unique(cutout[..., 0])[1:] # Remove 0 (background)
    return list(valid_ids)

@app.function(image=image, max_containers=10, timeout=300)
def fetch_and_clip_mesh(seg_id):
    """Step 2: Map function to fetch full-resolution meshes in parallel and apply surgical cuts."""
    import cloudvolume
    import pyvista as pv
    import tempfile
    
    vol = cloudvolume.CloudVolume('gs://h01-release/data/20210601/c3', use_https=True)
    
    try:
        seg_id = int(seg_id)
        # Download the full, un-approximated mesh
        mesh_data = vol.mesh.get(seg_id)[seg_id]
        vertices = mesh_data.vertices
        faces = mesh_data.faces
        
        if len(faces) == 0:
            return None
            
        # Convert to PyVista PolyData representation
        pv_faces = np.column_stack((np.full(len(faces), 3), faces)).flatten()
        pv_mesh = pv.PolyData(vertices, pv_faces)
        
        # Surgical Cut: Truncate the sweeping arbors strictly at the boundaries of our prism
        clipped = pv_mesh.clip_box(BOUNDS, invert=False)
        if clipped.n_points == 0:
            return None
            
        # Serialize the un-decimated geometry safely to pass between cloud instances
        with tempfile.NamedTemporaryFile(suffix='.vtu') as f:
            clipped.save(f.name)
            f.seek(0)
            return f.read()
    except Exception as e:
        print(f"Failed processing {seg_id}: {e}")
        # Some segments might not have precomputed meshes yet
        return None

@app.function(image=image, gpu="A100", memory=32768, timeout=600)
def assemble_and_render(vtp_bytes_list):
    """Step 3: Reduce function running on a heavy A100 to ingest all meshes and render at 4K."""
    import pyvista as pv
    import tempfile
    import os
    from PIL import Image

    print(f"Ingesting {len(vtp_bytes_list)} fully-resolved neuronal meshes...")
    
    # Initialize virtual framebuffer for headless GPU rendering
    os.system('Xvfb :99 -screen 0 3840x2160x24 > /dev/null 2>&1 &')
    os.environ['DISPLAY'] = ':99'
    
    plotter = pv.Plotter(off_screen=True, window_size=[3840, 2160])
    plotter.set_background('black')

    # Load and colorize each un-approximated mesh
    for vtp_bytes in vtp_bytes_list:
        with tempfile.NamedTemporaryFile(suffix='.vtu', delete=False) as f:
            f.write(vtp_bytes)
            f_name = f.name
            
        mesh = pv.read(f_name)
        color = tuple(np.random.rand(3))
        plotter.add_mesh(mesh, color=color, smooth_shading=True, opacity=1.0)
        os.remove(f_name)
        
    print("Constructing the structural bounding box...")
    box = pv.Box(bounds=BOUNDS)
    plotter.add_mesh(box, style='wireframe', color='white', line_width=4)
    
    plotter.camera_position = 'iso'
    
    print("Exporting massive interactive HTML WebGL scene...")
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
        html_path = f.name
    
    plotter.export_html(html_path)
    
    with open(html_path, 'r') as f:
        html_content = f.read()
        
    os.remove(html_path)
    return html_content

@app.local_entrypoint()
def main():
    print("--- INITIATING MASSIVE PARALLEL PIPELINE ---")
    
    ids = get_neuron_ids.remote()
    print(f"Identified {len(ids)} unique neurons in target prism.")
    print("Spawning ~10 parallel workers to stream and surgically cut meshes...")
    
    # Map execution across the Modal cluster
    vtp_data = list(fetch_and_clip_mesh.map(ids))
    valid_vtp = [d for d in vtp_data if d is not None]
    
    print(f"Successfully processed {len(valid_vtp)} meshes in record time.")
    print("Dispatching heavy assembly to A100 Tensor Core GPU...")
    
    # Render execution
    html_string = assemble_and_render.remote(valid_vtp)
    
    output_filename = "h01_interactive_no_approximation.html"
    with open(output_filename, "w") as f:
        f.write(html_string)
        
    print(f"--- SUCCESS: Spectacular interactive render saved to {output_filename} ---")
