# Embodiment

A generative, interactive artwork exploring the intersection of biological connectomics and human kinematics.

## Overview

This repository hosts a minimal, highly optimized WebGL interactive scene. It presents a dynamic $3\times3\times3$ spatial matrix of volumetric hand models, programmatically extracted and aligned using the HaMeR (Hand Mesh Recovery) neural architecture.

The geometry is painted with a cinematic, procedurally generated color palette derived from spatial k-means clustering. The interactive layout rotates reactively to user input, blending rigorous engineering with minimalist aesthetic presentation.

## Usage

Start a local server to bypass CORS policies for `.glb` loading:

```bash
python3 -m http.server 8000
```

Navigate to `http://localhost:8000/index.html` in your web browser.

## Pipeline

The geometry and palette were generated via a distributed Modal cloud pipeline:
- `scripts/extract_palette.py`: Extracts cinematic color gradients from source video using scikit-learn.
- `scripts/extract_hands.py`: Reconstructs 3D volumetric hand meshes from video frames via HaMeR.
- `scripts/generate_scene.py`: Compiles and colors the spatial grid matrix, exporting the final `.glb` artifact.

## License

Open-sourced under the MIT License.
