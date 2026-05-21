# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "modal",
# ]
# ///

import modal
import os
import sys

CONCURRENCY = 10 if any(arg == "--parallel-10x" or arg.startswith("--parallel-10x=") for arg in sys.argv) else 1


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "git")
    .uv_pip_install(
        "pip",
        "setuptools<70",
        "torch>=2.0.0", 
        "torchvision", 
        "opencv-python-headless", 
        "numpy", 
        "tqdm",
        "huggingface_hub"
    )
)

app = modal.App("sapiens-feature-extraction")

def depth_to_mesh_obj(depth_map, mask):
    """Generates a closed watertight OBJ string from a depth map and binary mask."""
    import numpy as np
    import cv2
    
    h, w = depth_map.shape
    
    # Erode mask to remove glitchy edges where depth falls off into the background
    kernel = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    
    # Normalized coordinates
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    x_norm = (x_grid - w / 2.0) / w
    y_norm = (y_grid - h / 2.0) / w
    
    # Valid pixels
    valid = eroded_mask > 0
    
    vertex_count = np.count_nonzero(valid)
    if vertex_count == 0:
        return b""

    # Map from (y, x) to vertex index (1-based for OBJ)
    vertex_indices = np.zeros((h, w), dtype=np.int32)
    vertex_indices[valid] = np.arange(1, vertex_count + 1)
    
    # Create front vertices (Z is mapped to negative to go into the screen)
    x_valid = x_norm[valid]
    y_valid = y_norm[valid]
    z_valid = depth_map[valid]
    front_vertices = np.stack([x_valid, -y_valid, -z_valid], axis=1)
    
    # Create back vertices at a fixed flat plane (Z=0.0)
    back_vertices = np.stack([x_valid, -y_valid, np.zeros_like(z_valid)], axis=1)
    
    # Total vertices
    vertices = np.vstack([front_vertices, back_vertices])
    
    # Shift arrays to get neighbors
    v00 = vertex_indices[:-1, :-1]
    v01 = vertex_indices[:-1, 1:]
    v10 = vertex_indices[1:, :-1]
    v11 = vertex_indices[1:, 1:]
    
    # Triangle 1: (r, c), (r+1, c), (r, c+1) -> v00, v10, v01
    valid_t1 = (v00 > 0) & (v10 > 0) & (v01 > 0)
    
    # Triangle 2: (r+1, c), (r+1, c+1), (r, c+1) -> v10, v11, v01
    valid_t2 = (v10 > 0) & (v11 > 0) & (v01 > 0)
    
    t1_faces = np.stack([v00[valid_t1], v10[valid_t1], v01[valid_t1]], axis=1)
    t2_faces = np.stack([v10[valid_t2], v11[valid_t2], v01[valid_t2]], axis=1)
    front_faces = np.vstack([t1_faces, t2_faces]) if len(t1_faces) > 0 or len(t2_faces) > 0 else np.empty((0, 3), dtype=np.int32)
    
    # Create back faces (reversed winding order)
    if len(front_faces) > 0:
        back_faces = front_faces[:, [0, 2, 1]] + vertex_count
    else:
        back_faces = np.empty((0, 3), dtype=np.int32)
        
    # Find boundary edges to create walls
    if len(front_faces) > 0:
        e1 = front_faces[:, [0, 1]]
        e2 = front_faces[:, [1, 2]]
        e3 = front_faces[:, [2, 0]]
        edges = np.vstack([e1, e2, e3])
        
        base = np.int64(vertex_count + 2)
        edge_ids = edges[:, 0].astype(np.int64) * base + edges[:, 1].astype(np.int64)
        reverse_edge_ids = edges[:, 1].astype(np.int64) * base + edges[:, 0].astype(np.int64)
        
        is_boundary = ~np.isin(edge_ids, reverse_edge_ids)
        boundary_edges = edges[is_boundary]
        
        vA = boundary_edges[:, 0]
        vB = boundary_edges[:, 1]
        vA_back = vA + vertex_count
        vB_back = vB + vertex_count
        
        wall_t1 = np.stack([vA, vB, vB_back], axis=1)
        wall_t2 = np.stack([vA, vB_back, vA_back], axis=1)
        wall_faces = np.vstack([wall_t1, wall_t2])
    else:
        wall_faces = np.empty((0, 3), dtype=np.int32)
        
    all_faces = np.vstack([front_faces, back_faces, wall_faces]) if len(front_faces) > 0 else np.empty((0, 3), dtype=np.int32)
    
    # Build obj string
    obj_lines = []
    for v in vertices:
        obj_lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}")
    
    for f in all_faces:
        obj_lines.append(f"f {f[0]} {f[1]} {f[2]}")
        
    return "\n".join(obj_lines).encode('utf-8')


@app.function(image=image, timeout=600)
def extract_video_frames(video_bytes: bytes, frame_count: int) -> list[tuple[int, bytes]]:
    import tempfile
    import cv2
    import os
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
        tmp_video.write(video_bytes)
        tmp_video_path = tmp_video.name

    cap = cv2.VideoCapture(tmp_video_path)
    
    frames = []
    idx = 0
    while cap.isOpened():
        if frame_count > 0 and idx >= frame_count:
            break
        success, img = cap.read()
        if not success:
            break
            
        success, img_encoded = cv2.imencode('.jpg', img)
        if success:
            frames.append((idx, img_encoded.tobytes()))
        idx += 1
        
    cap.release()
    os.remove(tmp_video_path)
    return frames


@app.cls(image=image, gpu="A100", timeout=1200, max_containers=CONCURRENCY)
class SapiensPredictor:
    @modal.enter()
    def load_models(self):
        import torch
        from huggingface_hub import hf_hub_download
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading Sapiens 1B/2B models on {self.device}...")
        
        depth_ckpt = hf_hub_download(repo_id='facebook/sapiens-depth-2b-torchscript', filename='sapiens_2b_render_people_epoch_25_torchscript.pt2')
        normal_ckpt = hf_hub_download(repo_id='facebook/sapiens-normal-2b-torchscript', filename='sapiens_2b_normal_render_people_epoch_70_torchscript.pt2')
        seg_ckpt = hf_hub_download(repo_id='facebook/sapiens-seg-1b-torchscript', filename='sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2')
        
        self.model_depth = torch.jit.load(depth_ckpt).to(self.device)
        self.model_normal = torch.jit.load(normal_ckpt).to(self.device)
        self.model_seg = torch.jit.load(seg_ckpt).to(self.device)

    @modal.method()
    def process_frame(self, frame_data: tuple[int, bytes]):
        import cv2
        import torch
        import torch.nn.functional as F
        import numpy as np

        frame_idx, img_bytes = frame_data
        
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        w = img.shape[1]
        h = img.shape[0]
        
        # Center crop to 3:4 aspect ratio (width:height) -> 768:1024
        target_aspect = 768.0 / 1024.0
        img_aspect = w / h
        
        if img_aspect > target_aspect:
            new_w = int(h * target_aspect)
            start_x = (w - new_w) // 2
            crop = img[:, start_x:start_x+new_w]
        else:
            new_h = int(w / target_aspect)
            start_y = (h - new_h) // 2
            crop = img[start_y:start_y+new_h, :]
            
        crop_resized = cv2.resize(crop, (768, 1024), interpolation=cv2.INTER_LINEAR)
        
        # Preprocess
        img_tensor = crop_resized.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img_tensor)[[2, 1, 0], ...].float()
        mean = torch.tensor([123.5, 116.5, 103.5]).view(-1, 1, 1)
        std = torch.tensor([58.5, 57.0, 57.5]).view(-1, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        batch_img = img_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            res_seg = self.model_seg(batch_img)
            seg_logits = F.interpolate(res_seg[0].unsqueeze(0), size=(1024, 768), mode="bilinear").squeeze(0)
            pred_sem_seg = seg_logits.argmax(dim=0).cpu().numpy()
            mask = (pred_sem_seg > 0).astype(np.uint8) * 255
            
            res_depth = self.model_depth(batch_img)
            depth_logits = F.interpolate(res_depth[0].unsqueeze(0), size=(1024, 768), mode="bilinear").squeeze(0)
            depth_map = depth_logits[0].cpu().numpy()
            
            res_normal = self.model_normal(batch_img)
            normal_logits = F.interpolate(res_normal[0].unsqueeze(0), size=(1024, 768), mode="bilinear").squeeze(0)
            normal_map = normal_logits.float().cpu().numpy().transpose(1, 2, 0)
        
        # Post-process Depth
        depth_map[mask == 0] = np.nan
        min_val = np.nanmin(depth_map)
        max_val = np.nanmax(depth_map)
        if np.isnan(min_val) or np.isnan(max_val) or max_val == min_val:
            depth_normalized = np.zeros_like(depth_map)
        else:
            depth_normalized = 1.0 - ((depth_map - min_val) / (max_val - min_val))
            depth_normalized[np.isnan(depth_normalized)] = 0.0
        
        depth_16bit = (depth_normalized * 65535.0).astype(np.uint16)
        
        # Post-process Normal
        normal_map_norm = np.linalg.norm(normal_map, axis=-1, keepdims=True)
        normal_map_normalized = normal_map / (normal_map_norm + 1e-5)
        normal_map_normalized[mask == 0] = -1
        normal_img = ((normal_map_normalized + 1) / 2 * 255).astype(np.uint8)
        normal_img = normal_img[:, :, ::-1] # RGB to BGR for cv2
        
        # Mesh Generation
        mesh_bytes = depth_to_mesh_obj(depth_normalized, mask)
        
        # Encode images
        _, depth_png = cv2.imencode('.png', depth_16bit)
        _, normal_png = cv2.imencode('.png', normal_img)
        _, mask_png = cv2.imencode('.png', mask)
        
        return frame_idx, depth_png.tobytes(), normal_png.tobytes(), mask_png.tobytes(), mesh_bytes

@app.local_entrypoint()
def main(video_path: str = "input_hand_video.mp4", out_dir: str = "sapiens_cache", frame_count: int = 0, parallel_10x: bool = False):
    import os
    
    if not os.path.exists(video_path):
        print(f"Error: Input video '{video_path}' not found.")
        return
        
    print(f"Reading {video_path}...")
    with open(video_path, "rb") as f:
        video_bytes = f.read()
        
    print("Extracting frames on Modal...")
    frames = extract_video_frames.remote(video_bytes, frame_count)
    
    if not frames:
        print("No frames extracted.")
        return
        
    print(f"Extracted {len(frames)} frames. Launching parallel Sapiens inference on Modal...")
    
    predictor = SapiensPredictor()
    
    # Process frames in parallel
    results = predictor.process_frame.map(frames)
    
    print(f"Received all results. Writing to '{out_dir}'...")
    os.makedirs(out_dir, exist_ok=True)
    
    for frame_idx, depth_bytes, normal_bytes, mask_bytes, mesh_bytes in results:
        with open(os.path.join(out_dir, f"depth_{frame_idx:04d}.png"), "wb") as f:
            f.write(depth_bytes)
        with open(os.path.join(out_dir, f"normal_{frame_idx:04d}.png"), "wb") as f:
            f.write(normal_bytes)
        with open(os.path.join(out_dir, f"mask_{frame_idx:04d}.png"), "wb") as f:
            f.write(mask_bytes)
        if mesh_bytes:
            with open(os.path.join(out_dir, f"mesh_{frame_idx:04d}.obj"), "wb") as f:
                f.write(mesh_bytes)
            
    print(f"Successfully extracted frames and generated meshes to {out_dir}/")
