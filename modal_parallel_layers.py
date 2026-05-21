import modal
import io
import numpy as np
import tempfile

image = (
    modal.Image.debian_slim()
    .apt_install("xvfb", "libgl1-mesa-glx", "libglib2.0-0")
    .uv_pip_install("cloud-volume", "numpy", "pyvista[jupyter]", "pillow", "imageio[ffmpeg]")
)

app = modal.App("h01-parallel-layers-artifical")

# H01 Resolution: 8x8x33 nm
X_RES, Y_RES, Z_RES = 8, 8, 33

X_START, X_END = 150000, 150500
Z_START, Z_END = 2000, 2060

# We sample widely separated Y coordinates (depth) to genuinely sample different anatomical layers
# We sample distinct Y coordinates (depth) near the center of the sample wedge
Y_SAMPLES = [150000, 151000, 152000, 153000, 154000, 155000]
Y_THICKNESS = 500

@app.function(image=image, memory=4096)
def get_neuron_ids():
    """Step 1: Fetch IDs from the 6 separated anatomical samples."""
    import cloudvolume
    vol = cloudvolume.CloudVolume('gs://h01-release/data/20210601/c3', use_https=True)
    
    tasks = []
    for layer_idx, y_start in enumerate(Y_SAMPLES):
        y_end = y_start + Y_THICKNESS
        try:
            cutout = vol[X_START:X_END, y_start:y_end, Z_START:Z_END]
            valid_ids = np.unique(cutout[..., 0])[1:]
            
            # Create a task for each unique ID, including the layer info
            for seg_id in valid_ids:
                tasks.append({
                    "seg_id": seg_id,
                    "layer_idx": layer_idx,
                    "y_start": y_start,
                    "y_end": y_end
                })
        except Exception as e:
            print(f"Skipping empty volume slice at Y={y_start}: {e}")
            continue
    return tasks

@app.function(image=image, max_containers=10, timeout=300)
def fetch_and_clip_mesh(task):
    """Step 2: Map function to fetch full-resolution meshes and clip them to their true anatomical bounding box."""
    import cloudvolume
    import pyvista as pv
    import tempfile
    
    seg_id = task["seg_id"]
    y_start = task["y_start"]
    y_end = task["y_end"]
    layer_idx = task["layer_idx"]
    
    vol = cloudvolume.CloudVolume('gs://h01-release/data/20210601/c3', use_https=True)
    
    try:
        seg_id = int(seg_id)
        mesh_data = vol.mesh.get(seg_id)[seg_id]
        vertices = mesh_data.vertices
        faces = mesh_data.faces
        
        if len(faces) == 0:
            return None
            
        pv_faces = np.column_stack((np.full(len(faces), 3), faces)).flatten()
        pv_mesh = pv.PolyData(vertices, pv_faces)
        
        # Surgical Cut to their true anatomical bounds
        true_bounds = [
            X_START*X_RES, X_END*X_RES, 
            y_start*Y_RES, y_end*Y_RES, 
            Z_START*Z_RES, Z_END*Z_RES
        ]
        clipped = pv_mesh.clip_box(true_bounds, invert=False)
        if clipped.n_points == 0:
            return None
            
        with tempfile.NamedTemporaryFile(suffix='.vtu') as f:
            clipped.save(f.name)
            f.seek(0)
            return {"vtu_bytes": f.read(), "layer_idx": layer_idx, "y_start": y_start}
    except Exception as e:
        return None

@app.function(image=image, gpu="A100", memory=32768, timeout=600)
def assemble_and_render(mesh_tasks):
    """Step 3: Reduce function to assemble, translate, artificially stack, and render."""
    import pyvista as pv
    import tempfile
    import os

    print(f"Ingesting {len(mesh_tasks)} fully-resolved neuronal meshes...")
    
    os.system('Xvfb :99 -screen 0 3840x2160x24 > /dev/null 2>&1 &')
    os.environ['DISPLAY'] = ':99'
    
    plotter = pv.Plotter(off_screen=True, window_size=[3840, 2160])
    plotter.set_background('black')

    layer_names = ["I", "II", "III", "IV", "V", "VI"]
    
    BASE_Y = 0
    STACK_Y_OFFSET = Y_THICKNESS * Y_RES * 1.5 
    
    for layer_idx in range(6):
        artifical_y_start = BASE_Y + layer_idx * STACK_Y_OFFSET
        artifical_y_end = artifical_y_start + Y_THICKNESS * Y_RES
        
        stacked_bounds = [
            X_START * X_RES, X_END * X_RES,
            artifical_y_start, artifical_y_end,
            Z_START * Z_RES, Z_END * Z_RES
        ]
        
        box = pv.Box(bounds=stacked_bounds)
        plotter.add_mesh(box, style='wireframe', color='white', line_width=1, opacity=0.3)
        
        # Add large 3D serif text aligned slightly off the horizontal
        text_mesh = pv.Text3D(layer_names[layer_idx], depth=0.1)
        text_mesh.scale([2000, 2000, 2000], inplace=True)
        # Tilt off horizontal
        text_mesh.rotate_z(15, inplace=True)
        text_mesh.rotate_y(-10, inplace=True)
        
        # Position slightly to the side of the box
        x_center = X_START * X_RES - 4000
        y_center = (artifical_y_start + artifical_y_end) / 2 - 1000
        z_center = (Z_START * Z_RES + Z_END * Z_RES) / 2
        text_mesh.translate([x_center, y_center, z_center], inplace=True)
        
        plotter.add_mesh(text_mesh, color='white')

    # Add all translated neurons
    for task in mesh_tasks:
        with tempfile.NamedTemporaryFile(suffix='.vtu', delete=False) as f:
            f.write(task["vtu_bytes"])
            f_name = f.name
            
        mesh = pv.read(f_name)
        
        layer_idx = task["layer_idx"]
        original_y_start = task["y_start"] * Y_RES
        target_y_start = BASE_Y + layer_idx * STACK_Y_OFFSET
        y_translation = target_y_start - original_y_start
        
        mesh.translate([0, y_translation, 0], inplace=True)
        
        color = tuple(np.random.rand(3))
        plotter.add_mesh(mesh, color=color, smooth_shading=True, opacity=1.0)
        os.remove(f_name)

    plotter.camera_position = 'iso'
    
    print("Setting up 360-degree rotation animation...")
    movie_path = os.path.join(tempfile.gettempdir(), 'render.mp4')
    plotter.open_movie(movie_path, framerate=30)
    plotter.show(auto_close=False, interactive=False)
    
    frames = 120
    for i in range(frames):
        plotter.camera.Azimuth(360.0 / frames)
        plotter.write_frame()
        
    plotter.close()
    
    with open(movie_path, 'rb') as f:
        mp4_content = f.read()
        
    os.remove(movie_path)
    return mp4_content

@app.local_entrypoint()
def main():
    print("--- INITIATING ARTIFICIAL STACK PIPELINE ---")
    
    tasks = get_neuron_ids.remote()
    print(f"Sampled {len(tasks)} unique neurons across 6 distant anatomical depths.")
    print("Spawning ~10 parallel workers to stream and surgically cut meshes...")
    
    mesh_results = list(fetch_and_clip_mesh.map(tasks))
    valid_meshes = [d for d in mesh_results if d is not None]
    
    print(f"Successfully processed {len(valid_meshes)} meshes.")
    print("Dispatching A100 to translate, stack, and render 360 video...")
    
    mp4_bytes = assemble_and_render.remote(valid_meshes)
    
    output_filename = "h01_artificially_stacked.mp4"
    with open(output_filename, "wb") as f:
        f.write(mp4_bytes)
        
    print(f"--- SUCCESS: Render saved to {output_filename} ---")
