import modal
import numpy as np
import trimesh
import os

image = (
    modal.Image.debian_slim()
    .uv_pip_install("numpy", "trimesh", "scipy")
    .add_local_dir("hamer_cache", remote_path="/hamer_cache")
    .add_local_file("v1/palette.json", remote_path="/palette.json")
)

app = modal.App("multihand-scene")

@app.function(image=image, memory=4096, timeout=600)
def build_scene():
    from scipy.spatial.transform import Rotation
    
    scene = trimesh.Scene()
    
    print("Arranging hands in 3x3x3 grid...")
    hand_files = sorted([f for f in os.listdir("/hamer_cache") if f.endswith(".ply")])
    print(f"Found {len(hand_files)} hand files in /hamer_cache.")
    
    grid_coords = np.linspace(-35, 35, 3)
    hand_idx = 0
    
    print("Loading palette...")
    import json
    with open("/palette.json", "r") as f:
        hex_colors = json.load(f)
        
    palette_rgba = []
    for h in hex_colors:
        h = h.lstrip('#')
        rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        palette_rgba.append([rgb[0], rgb[1], rgb[2], 255])
    
    for x in grid_coords:
        for y in grid_coords:
            for z in grid_coords:
                if hand_idx >= len(hand_files):
                    break
                    
                file_path = os.path.join("/hamer_cache", hand_files[hand_idx])
                hand_mesh = trimesh.load(file_path)
                
                hand_mesh.vertices -= hand_mesh.centroid
                
                size = np.max(hand_mesh.extents)
                scale = 20.0 / size if size > 0 else 1.0
                hand_mesh.vertices *= scale
                
                color = palette_rgba[hand_idx % len(palette_rgba)]
                hand_mesh.visual.vertex_colors = np.tile(color, (len(hand_mesh.vertices), 1))
                
                translation = np.eye(4)
                translation[:3, 3] = [x, y, z]
                
                rot = np.eye(4)
                rot[:3, :3] = Rotation.random().as_matrix()
                
                transform = translation @ rot
                
                scene.add_geometry(hand_mesh, node_name=f"hand_{hand_idx}", transform=transform)
                hand_idx += 1

    print("Exporting GLB to /tmp/scene.glb...")
    scene.export("/tmp/scene.glb")
    
    print("Streaming GLB back to client...")
    with open("/tmp/scene.glb", "rb") as f:
        while chunk := f.read(1024 * 1024 * 2): # 2MB chunks
            yield chunk

@app.local_entrypoint()
def main():
    print("Starting remote build_scene generator...")
    with open("scene.glb", "wb") as f:
        for chunk in build_scene.remote_gen():
            f.write(chunk)
    print("Successfully downloaded scene.glb!")
