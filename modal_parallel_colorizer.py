import modal
import io
import numpy as np
import json

image = (
    modal.Image.debian_slim()
    .apt_install("xvfb", "libgl1-mesa-glx", "libglib2.0-0", "blender", "libegl1", "ffmpeg")
    .uv_pip_install("cloud-volume", "numpy", "pyvista[jupyter]", "pillow", "scipy")
)

app = modal.App("h01-parallel-render-colorized")
volume = modal.Volume.from_name("neuron-render-vol", create_if_missing=True)

def compute_spherical_morph(vertices, L, R, t):
    t = np.clip(t, 0.0, 1.0)
    t_ease = 3.0 * (t**2) - 2.0 * (t**3)
    
    if t_ease == 0.0:
        return vertices.copy()

    x = vertices[:, 0] / L
    y = vertices[:, 1] / L
    z = vertices[:, 2] / L
    
    x = np.clip(x, -1.0, 1.0)
    y = np.clip(y, -1.0, 1.0)
    z = np.clip(z, -1.0, 1.0)
    
    x2 = x**2
    y2 = y**2
    z2 = z**2
    
    xs = x * np.sqrt(1.0 - (y2 / 2.0) - (z2 / 2.0) + (y2 * z2 / 3.0))
    ys = y * np.sqrt(1.0 - (x2 / 2.0) - (z2 / 2.0) + (x2 * z2 / 3.0))
    zs = z * np.sqrt(1.0 - (x2 / 2.0) - (y2 / 2.0) + (x2 * y2 / 3.0))
    
    S = np.column_stack((xs, ys, zs))
    S *= R 
    
    if t_ease == 1.0:
        return S
        
    return (1.0 - t_ease) * vertices + t_ease * S

@app.function(image=image, memory=4096)
def get_neuron_library():
    import cloudvolume
    vol = cloudvolume.CloudVolume('gs://h01-release/data/20210601/c3', use_https=True)
    cutout = vol[150000:150400, 150000:150400, 2000:2100]
    valid_ids = np.unique(cutout[..., 0])[1:]
    return list(valid_ids)

@app.function(image=image, max_containers=50, timeout=600)
def fetch_and_smooth_mesh(seg_id):
    import cloudvolume
    import pyvista as pv
    import tempfile
    import numpy as np
    
    vol = cloudvolume.CloudVolume('gs://h01-release/data/20210601/c3', use_https=True)
    try:
        seg_id = int(seg_id)
        mesh_data = vol.mesh.get(seg_id)[seg_id]
        vertices, faces = mesh_data.vertices, mesh_data.faces
        if len(faces) == 0:
            return None
            
        pv_faces = np.column_stack((np.full(len(faces), 3), faces)).flatten()
        pv_mesh = pv.PolyData(vertices, pv_faces)
        pv_mesh = pv_mesh.smooth_taubin(n_iter=200)
        
        with tempfile.NamedTemporaryFile(suffix='.vtp') as f:
            pv_mesh.save(f.name)
            f.seek(0)
            return f.read()
    except Exception as e:
        return None

@app.function(image=image, gpu="A100", memory=32768, timeout=600, volumes={"/data": volume})
def prepare_geometry(vtp_bytes_list, palette):
    import pyvista as pv
    import tempfile
    import os
    import numpy as np

    print("Initializing A100 Geometry Preparation...")
    filtered_palette = [c for c in palette if c.lower() != '#000000']
    meshes, all_points = [], []
    
    for idx, vtp_bytes in enumerate(vtp_bytes_list):
        with tempfile.NamedTemporaryFile(suffix='.vtp', delete=False) as f:
            f.write(vtp_bytes)
            f_name = f.name
            
        mesh = pv.read(f_name)
        os.remove(f_name)
        
        color_hex = filtered_palette[idx % len(filtered_palette)]
        r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
        mesh.point_data['RGB'] = np.tile([r, g, b], (mesh.n_points, 1)).astype(np.uint8)
        
        meshes.append(mesh)
        all_points.append(mesh.points)
        
    massive_point_cloud = np.vstack(all_points)
    global_center = np.mean(massive_point_cloud, axis=0)
    
    for mesh in meshes:
        mesh.points = mesh.points - global_center

    print("Merging meshes...")
    merged = meshes[0].merge(meshes[1:])
    
    L = np.max(np.abs(merged.points))
    # Normalize globally so scale is precisely 1.0
    merged.points *= (1.0 / L)
    merged.active_scalars_name = 'RGB'
    
    merged.save("/data/base.ply")
    volume.commit()
    return L

@app.function(image=image, gpu="A100", memory=16384, max_containers=60, timeout=600, volumes={"/data": volume})
def render_frame(frame_idx):
    import pyvista as pv
    import numpy as np
    import os
    
    # Headless X11 context for Blender Eevee
    os.system('Xvfb :99 -screen 0 1920x1080x24 > /dev/null 2>&1 &')
    os.environ['DISPLAY'] = ':99'
    
    # Reload volume to ensure we see prepare_geometry's write
    volume.reload()
    
    mesh = pv.read("/data/base.ply")
    base_vertices = mesh.points.copy()
    
    # Vertices are pre-normalized, so bounds are strictly [-1, 1]
    t = frame_idx / 119.0
    morphed_points = compute_spherical_morph(base_vertices, L=1.0, R=1.0, t=t)
    
    with open(f"frame.bin", "wb") as f:
        f.write(morphed_points.astype(np.float32).tobytes())
        
    blender_script = f"""
import bpy
import os
import array
import math

bpy.ops.wm.read_factory_settings(use_empty=True)
world = bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)

bpy.ops.import_mesh.ply(filepath="/data/base.ply")
obj = bpy.context.active_object
mesh = obj.data

mat = bpy.data.materials.new(name="NeuronMat")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()

node_output = nodes.new(type='ShaderNodeOutputMaterial')
node_emission = nodes.new(type='ShaderNodeEmission')
node_vcol = nodes.new(type='ShaderNodeVertexColor')

has_vcol = False
if hasattr(mesh, "color_attributes") and len(mesh.color_attributes) > 0:
    node_vcol.layer_name = mesh.color_attributes[0].name
    has_vcol = True
elif hasattr(mesh, "vertex_colors") and len(mesh.vertex_colors) > 0:
    node_vcol.layer_name = mesh.vertex_colors[0].name
    has_vcol = True

if has_vcol:
    links.new(node_vcol.outputs['Color'], node_emission.inputs['Color'])
else:
    node_emission.inputs['Color'].default_value = (0.0, 0.8, 1.0, 1.0)

links.new(node_emission.outputs['Emission'], node_output.inputs['Surface'])
node_emission.inputs['Strength'].default_value = 2.0
obj.data.materials.append(mat)

with open("frame.bin", "rb") as f:
    a = array.array('f')
    a.fromfile(f, len(mesh.vertices) * 3)
    mesh.vertices.foreach_set("co", a)
    mesh.update()

cam_data = bpy.data.cameras.new('camera')
cam = bpy.data.objects.new('camera', cam_data)
bpy.context.collection.objects.link(cam)
bpy.context.scene.camera = cam

cam.data.clip_start = 0.01
cam.data.clip_end = 100.0
cam.location = (0, -2.5, 0)
cam.rotation_euler = (math.radians(90), 0, 0)

empty = bpy.data.objects.new("Empty", None)
bpy.context.collection.objects.link(empty)
cam.parent = empty

orbital_angle = math.radians(90) * ({frame_idx} / 119.0)
empty.rotation_euler = (0, 0, orbital_angle)

scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.filepath = os.path.abspath("frame_output.png")

scene.eevee.use_bloom = True
scene.eevee.bloom_intensity = 0.05
scene.eevee.taa_render_samples = 16

bpy.ops.render.render(write_still=True)
"""
    with open("render.py", "w") as f:
        f.write(blender_script)
        
    os.system("blender -b -P render.py")
    
    with open("frame_output.png", "rb") as f:
        return frame_idx, f.read()

@app.function(image=image, memory=4096)
def compile_video(frames_data):
    import os
    import subprocess
    
    print("Stitching GPU frames using FFmpeg...")
    os.makedirs("frames", exist_ok=True)
    for idx, png_bytes in frames_data:
        with open(f"frames/frame_{idx:03d}.png", "wb") as f:
            f.write(png_bytes)
            
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "30",
        "-i", "frames/frame_%03d.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "output.mp4"
    ], check=True)
    
    with open("output.mp4", "rb") as f:
        return f.read()

@app.local_entrypoint()
def main(palette_json: str):
    palette = json.loads(palette_json)
    print("--- INITIATING MASSIVE MULTI-GPU ANIMATION PIPELINE ---")
    
    ids = get_neuron_library.remote()
    vtp_data = list(fetch_and_smooth_mesh.map(ids))
    valid_vtp = [d for d in vtp_data if d is not None]
    
    print(f"Successfully processed {len(valid_vtp)} meshes. Preparing global topology...")
    L = prepare_geometry.remote(valid_vtp, palette)
    
    print(f"Dispatching MASSIVE MULTI-GPU FLEET for 120 frames (Base L={L:.2f})...")
    frames_input = range(120)
    results = list(render_frame.map(frames_input))
    
    print("All GPUs have completed rendering! Compiling H.264 video...")
    video_bytes = compile_video.remote(results)
    
    output_filename = "morph_animation.mp4"
    with open(output_filename, "wb") as f:
        f.write(video_bytes)
        
    print(f"--- SUCCESS: Spectacular Distributed Animation saved to {output_filename} ---")
