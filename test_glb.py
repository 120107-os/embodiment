import trimesh
import sys
try:
    print("Loading scene.glb...")
    scene = trimesh.load('scene.glb')
    print("Success! Scene loaded.")
    print(f"Geometries: {len(scene.geometry)}")
except Exception as e:
    print(f"Failed to load: {e}")
    sys.exit(1)
