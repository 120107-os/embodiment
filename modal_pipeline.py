import modal
import io
import numpy as np
import json
import os
import pyvista as pv
import tempfile

image = (
    modal.Image.debian_slim()
    .apt_install("xvfb", "libgl1-mesa-glx", "libglib2.0-0")
    .uv_pip_install("cloud-volume", "numpy", "pyvista[all,jupyter]", "trimesh", "trame", "trame-vtk", "trame-vuetify")
)

app = modal.App("hand-connectomics-pipeline")

# Massive physical region (5000x5000x500 base resolution)
X_RES, Y_RES, Z_RES = 8, 8, 33
MIN_X, MAX_X = 145000*X_RES, 150000*X_RES
MIN_Y, MAX_Y = 145000*Y_RES, 150000*Y_RES
MIN_Z, MAX_Z = 2000*Z_RES, 2500*Z_RES

NEURON_CENTER = np.array([(MIN_X + MAX_X)/2, (MIN_Y + MAX_Y)/2, (MIN_Z + MAX_Z)/2])
NEURON_SIZE = max(MAX_X - MIN_X, MAX_Y - MIN_Y, MAX_Z - MIN_Z)

@app.function(image=image, memory=8192)
def get_neuron_ids(n: int = 150):
    import cloudvolume
    import numpy as np
    
    # Query at mip=3 to load a massive region efficiently
    vol = cloudvolume.CloudVolume(
        'gs://h01-release/data/20210601/c3',
        use_https=True,
        mip=3
    )
    
    # mip=3 downsamples X and Y by 8, Z by 2
    bounds = (slice(145000//8, 150000//8), slice(145000//8, 150000//8), slice(2000//2, 2500//2))
    print(f"Querying volume bounds at mip=3: {bounds}")
    cutout = vol[bounds]
    
    # Compute voxel volume histogram
    unique_ids, counts = np.unique(cutout, return_counts=True)
    
    # Sort by voxel count descending to find largest structures
    sorted_indices = np.argsort(-counts)
    sorted_ids = unique_ids[sorted_indices]
    sorted_counts = counts[sorted_indices]
    
    valid_ids = []
    print(f"Top {n} densest neurons in region:")
    for uid, count in zip(sorted_ids, sorted_counts):
        if uid != 0:
            valid_ids.append(uid)
            print(f"  ID: {uid} | Voxels: {count}")
        if len(valid_ids) == n:
            break
            
    return valid_ids

@app.function(image=image, max_containers=15, timeout=600)
def fetch_scale_and_clip(seg_id, hand_obj_bytes, hand_center, scale, color):
    import cloudvolume
    import pyvista as pv
    import tempfile
    
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
        
        # Scale and translate to hand
        pts = pv_mesh.points - NEURON_CENTER
        pts = pts * scale
        pts = pts + hand_center
        pv_mesh.points = pts
        
        # Load hand mesh for clipping
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            f.write(hand_obj_bytes)
            hand_path = f.name
            
        hand = pv.read(hand_path)
        
        # Perform the boolean geometric clip
        print(f"Clipping neuron {seg_id}...")
        print(f"Hand bounds: {hand.bounds}")
        print(f"Neuron bounds before clip: {pv_mesh.bounds}")
        
        # invert=True keeps inside
        clipped = pv_mesh.clip_surface(hand, invert=True)
        
        # If still empty, try select_enclosed_points
        if clipped.n_points == 0:
            print("clip_surface returned 0 points, trying select_enclosed_points...")
            enclosed = pv_mesh.select_enclosed_points(hand, check_surface=False)
            mask = enclosed['SelectedPoints'] == 1
            clipped = pv_mesh.extract_points(mask)
            
        print(f"Neuron {seg_id} points after clip: {clipped.n_points}")
        
        if clipped.n_points == 0:
            return None
            
        with tempfile.NamedTemporaryFile(suffix='.vtk', delete=False) as f:
            clipped.save(f.name)
            f.seek(0)
            return (f.read(), color)
            
    except Exception as e:
        print(f"Failed processing {seg_id}: {e}")
        return None

@app.function(image=image, gpu="T4", memory=16384, timeout=600)
def render_scene(vtp_data_list, hand_obj_bytes):
    import pyvista as pv
    import tempfile
    import os

    os.system('Xvfb :99 -screen 0 1920x1080x24 > /dev/null 2>&1 &')
    os.environ['DISPLAY'] = ':99'
    
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    plotter.set_background('black')

    print("Loading hand mesh into scene...")
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
        f.write(hand_obj_bytes)
        hand_path = f.name
        
    hand = pv.read(hand_path)
    plotter.add_mesh(hand, color="white", opacity=0.15, style="surface", smooth_shading=True)
    os.remove(hand_path)
    
    print(f"Ingesting {len(vtp_data_list)} clipped neuronal meshes...")
    for vtp_bytes, color in vtp_data_list:
        with tempfile.NamedTemporaryFile(suffix='.vtk', delete=False) as f:
            f.write(vtp_bytes)
            f_name = f.name
            
        mesh = pv.read(f_name)
        plotter.add_mesh(mesh, color=color, smooth_shading=True, specular=0.5, ambient=0.2)
        os.remove(f_name)
        
    plotter.camera_position = 'iso'
    
    print("Exporting interactive HTML scene...")
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
        html_path = f.name
    
    plotter.export_html(html_path)
    
    with open(html_path, 'r') as f:
        html_content = f.read()
        
    os.remove(html_path)
    return html_content

@app.local_entrypoint()
def main():
    print("--- STARTING MODAL HAND-CONNECTOMICS PIPELINE ---")
    
    # Load color palette
    palette = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#00ffff"]
    if os.path.exists("palette.json"):
        with open("palette.json", "r") as f:
            palette = json.load(f)
            
    # Load Hand Mesh locally to calculate scale
    hand_path = "hamer_cache/mesh_0000_right_1.ply"
    if not os.path.exists(hand_path):
        print(f"Error: {hand_path} not found. Run modal_hamer_extract.py first.")
        return
        
    print(f"Loading {hand_path} to calculate spatial transform...")
    hand = pv.read(hand_path)
    hand_bounds = hand.bounds
    hand_center = np.array([(hand_bounds[0] + hand_bounds[1])/2, 
                            (hand_bounds[2] + hand_bounds[3])/2, 
                            (hand_bounds[4] + hand_bounds[5])/2])
    hand_size = max(hand_bounds[1] - hand_bounds[0], 
                    hand_bounds[3] - hand_bounds[2], 
                    hand_bounds[5] - hand_bounds[4])
                    
    scale = (hand_size * 0.9) / NEURON_SIZE if NEURON_SIZE > 0 else 1.0
    print(f"Calculated scale factor: {scale:.6f}")
    
    with open(hand_path, "rb") as f:
        hand_obj_bytes = f.read()
        
    # 1. Fetch Segment IDs
    print("Fetching segment IDs...")
    ids = get_neuron_ids.remote(n=150)
    print(f"Selected {len(ids)} unique neurons. Launching parallel fetch and clip...")
    
    # Prepare arguments for map
    args = []
    for i, seg_id in enumerate(ids):
        color = palette[i % len(palette)]
        args.append((seg_id, hand_obj_bytes, hand_center, scale, color))
        
    # Map execution across the Modal cluster
    results = fetch_scale_and_clip.starmap(args)
    valid_data = [d for d in results if d is not None]
    
    print(f"Successfully clipped {len(valid_data)} neuronal meshes.")
    print("Dispatching rendering task to GPU...")
    
    html_string = render_scene.remote(valid_data, hand_obj_bytes)
    
    output_filename = "interactive_hand_connectomics.html"
    with open(output_filename, "w") as f:
        f.write(html_string)
        
    print(f"--- SUCCESS: Interactive render saved to {output_filename} ---")
