# Generative Art Portfolio

A scalable, multi-versioned repository hosting generative, interactive artworks. The structure is designed to support evolving iterations (`v1`, `v2`, etc.) while maintaining a stable, unified gallery.

## Exhibitions

- **[V1 - Interactive Spatial Grid]**: A dynamic $3\times3\times3$ spatial matrix of volumetric hand models, programmatically extracted and aligned using neural architectures.

## Usage

Start a local server from the root directory to bypass CORS policies for `.glb` loading:

```bash
python3 -m http.server 8000
```

Navigate to `http://localhost:8000/index.html` in your web browser to view the gallery portal.

## Pipeline Architecture

To support version control and prevent code duplication, the data pipelines are split:
- **Core Library (`/core/`)**: Shared, immutable deep-learning extractors (e.g., mesh tracking).
- **Versioned Scripts (`/v1/scripts/`)**: Art-specific procedural generators that compile the scenes and bake the cinematic palettes.

## License

Open-sourced under the MIT License.
