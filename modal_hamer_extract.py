# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "modal==1.4.3",
# ]
# ///

import modal
import os
import sys

CONCURRENCY = 10 if any(arg == "--parallel-10x" or arg.startswith("--parallel-10x=") for arg in sys.argv) else 1

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "git", "wget", "tar", "build-essential", "libegl1", "libegl1-mesa-dev")
    .run_commands("git clone --recursive https://github.com/geopavlakos/hamer.git /hamer")
    .workdir("/hamer")
    # Install PyTorch
    .run_commands("uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --system")
    # 1. Install pip, numpy < 2, and cython so unisolated builds succeed without NumPy 2.0 binary incompatibility
    .run_commands("uv pip install pip 'numpy<2.0.0' cython --system")
    # 2. Build chumpy and xtcocotools which lack proper pyproject.toml build-system declarations
    .run_commands("uv pip install chumpy xtcocotools --no-build-isolation --system")
    # 3. Build vendored ViTPose (mmpose 0.24.0) so it doesn't pull the breaking PyPI mmpose
    .run_commands("uv pip install --no-build-isolation -e third-party/ViTPose --system")
    # 4. Build HaMeR itself without isolation to respect previously installed dependencies
    .run_commands("uv pip install --no-build-isolation -e '.[all]' --system")
    # Additional required packages
    .uv_pip_install(
        "opencv-python-headless",
        "tqdm",
        "huggingface_hub",
        "scikit-image",
        "numpy<2.0.0",
        "trimesh",
        "pyrender"
    )
    # Download MANO model and checkpoint weights
    .run_commands(
        "wget https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz",
        "tar --warning=no-unknown-keyword --exclude='.*' -xvf hamer_demo_data.tar.gz",
        "rm hamer_demo_data.tar.gz",
        "mkdir -p _DATA/data/mano",
        "wget https://huggingface.co/spaces/geopavlakos/HaMeR/resolve/main/_DATA/data/mano/MANO_RIGHT.pkl -O _DATA/data/mano/MANO_RIGHT.pkl"
    )
)

app = modal.App("hamer-mano-extraction")

@app.function(image=image, timeout=600)
def extract_video_frames(video_bytes: bytes, frame_count: int) -> list[tuple[int, bytes]]:
    import tempfile
    import cv2
    import os
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
        tmp_video.write(video_bytes)
        tmp_video_path = tmp_video.name

    cap = cv2.VideoCapture(tmp_video_path)
    import numpy as np
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count > 0 and frame_count < total_frames:
        target_indices = set(np.linspace(0, total_frames - 1, frame_count, dtype=int))
    else:
        target_indices = set(range(total_frames))
    
    frames = []
    idx = 0
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
            
        if idx in target_indices:
            success, img_encoded = cv2.imencode('.jpg', img)
            if success:
                frames.append((idx, img_encoded.tobytes()))
        idx += 1
        
    cap.release()
    os.remove(tmp_video_path)
    return frames


@app.cls(image=image, gpu="A100", timeout=1200, max_containers=CONCURRENCY)
class HamerPredictor:
    @modal.enter()
    def load_models(self):
        import sys
        sys.path.append("/hamer")
        
        import os
        os.chdir("/hamer")
        
        import torch
        import os
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        from hamer.models import load_hamer, DEFAULT_CHECKPOINT
        from hamer.utils.renderer import Renderer
        from vitpose_model import ViTPoseModel
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        from detectron2.config import LazyConfig
        import hamer
        from pathlib import Path
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("Loading HaMeR and MANO models...")
        self.model, self.model_cfg = load_hamer(DEFAULT_CHECKPOINT)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print("Loading Detectron2 for body detection...")
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        
        print("Loading ViTPose for keypoint detection...")
        self.cpm = ViTPoseModel(self.device)
        
        # Renderer for extracting meshes
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)

    @modal.method()
    def process_frame(self, frame_data: tuple[int, bytes]):
        import sys
        sys.path.append("/hamer")
        import os
        os.chdir("/hamer")
        
        import cv2
        import torch
        import numpy as np
        import trimesh
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from hamer.utils import recursive_to
        from hamer.utils.renderer import cam_crop_to_full
        
        frame_idx, img_bytes = frame_data
        
        img_np = np.frombuffer(img_bytes, np.uint8)
        img_cv2 = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        img = img_cv2.copy()[:, :, ::-1] # BGR to RGB
        
        # 1. Detect humans in the image
        det_out = self.detector(img_cv2)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()
        
        if len(pred_bboxes) == 0:
            return frame_idx, []
            
        # 2. Detect keypoints to isolate hands
        vitposes_out = self.cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )
        
        bboxes = []
        is_right = []
        
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Left hand bounding box
            keyp = left_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(0)
                
            # Right hand bounding box
            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(1)
                
        if len(bboxes) == 0:
            return frame_idx, []
            
        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        
        # 3. Predict MANO parameters using HaMeR
        dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes, right, rescale_factor=2.0)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        
        meshes = []
        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model(batch)
                
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right_hand = bool(batch['right'][n].item())
                # Correct vertex x-axis for left hands
                verts[:,0] = (2*is_right_hand - 1) * verts[:,0]
                
                # Transform to camera space
                pred_cam = out['pred_cam']
                multiplier = (2*batch['right'] - 1)
                pred_cam[:,1] = multiplier * pred_cam[:,1]
                
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                
                pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
                cam_t = pred_cam_t_full[n]
                camera_translation = cam_t.copy()
                
                # Convert vertices & faces to Trimesh 
                tmesh = self.renderer.vertices_to_trimesh(verts, camera_translation, (1, 1, 1), is_right=is_right_hand)
                
                # Export watertight mesh to OBJ and PLY byte string
                obj_bytes = trimesh.exchange.obj.export_obj(tmesh).encode('utf-8')
                ply_bytes = trimesh.exchange.ply.export_ply(tmesh)
                
                person_id = int(batch['personid'][n])
                meshes.append((person_id, is_right_hand, obj_bytes, ply_bytes))
                
        return frame_idx, meshes

@app.local_entrypoint()
def main(video_path: str = "input_hand_video.mp4", out_dir: str = "hamer_cache", frame_count: int = 0, parallel_10x: bool = False):
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
        
    print(f"Extracted {len(frames)} frames. Launching parallel HaMeR MANO inference on Modal...")
    
    predictor = HamerPredictor()
    
    # Process frames in parallel dynamically across A100 containers
    results = predictor.process_frame.map(frames)
    
    print(f"Received all results. Writing to '{out_dir}'...")
    os.makedirs(out_dir, exist_ok=True)
    
    for frame_idx, meshes in results:
        for person_id, is_right_hand, obj_bytes, ply_bytes in meshes:
            hand_type = "right" if is_right_hand else "left"
            obj_filename = f"mesh_{frame_idx:04d}_{hand_type}_{person_id}.obj"
            ply_filename = f"mesh_{frame_idx:04d}_{hand_type}_{person_id}.ply"
            with open(os.path.join(out_dir, obj_filename), "wb") as f:
                f.write(obj_bytes)
            with open(os.path.join(out_dir, ply_filename), "wb") as f:
                f.write(ply_bytes)
            
    print(f"Successfully generated anatomically correct MANO meshes to {out_dir}/")
