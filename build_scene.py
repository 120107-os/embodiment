# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyvista",
#     "trimesh",
#     "numpy",
#     "scipy",
# ]
# ///

import pyvista as pv
import numpy as np
import json
import glob
import os

def load_palette(path="palette.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#00ffff", "#ff00ff"]

def main():
    palette = load_palette()
    
    # 1. Load Hand Mesh
    # hamer_cache/mesh_0000_right_1.obj
    hand_mesh_path = "hamer_cache/mesh_0000_right_1.obj"
    if not os.path.exists(hand_mesh_path):
        print(f"Error: {hand_mesh_path} not found.")
        return
        
    hand = pv.read(hand_mesh_path)
    
    # 2. Load Neuronal Meshes
    neuron_files = glob.glob("h01_meshes/*.obj")
    if not neuron_files:
        print("Error: No neuron meshes found in h01_meshes/")
        return
        
    print(f"Loaded {len(neuron_files)} neurons.")
    
    neurons = []
    for f in neuron_files:
        neurons.append(pv.read(f))
        
    # Calculate bounds of all neurons
    all_bounds = np.array([n.bounds for n in neurons])
    min_x, max_x = all_bounds[:,0].min(), all_bounds[:,1].max()
    min_y, max_y = all_bounds[:,2].min(), all_bounds[:,3].max()
    min_z, max_z = all_bounds[:,4].min(), all_bounds[:,5].max()
    
    # Center of neurons
    neuron_center = np.array([(min_x + max_x)/2, (min_y + max_y)/2, (min_z + max_z)/2])
    
    # Size of neurons
    neuron_size = max(max_x - min_x, max_y - min_y, max_z - min_z)
    
    # Hand center and size
    hand_bounds = hand.bounds
    hand_center = np.array([(hand_bounds[0] + hand_bounds[1])/2, 
                            (hand_bounds[2] + hand_bounds[3])/2, 
                            (hand_bounds[4] + hand_bounds[5])/2])
    hand_size = max(hand_bounds[1] - hand_bounds[0], 
                    hand_bounds[3] - hand_bounds[2], 
                    hand_bounds[5] - hand_bounds[4])
                    
    # Scale factor (make neurons slightly smaller than the hand)
    scale = (hand_size * 0.9) / neuron_size if neuron_size > 0 else 1.0
    
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("black")
    
    # Add glass hand
    plotter.add_mesh(hand, color="white", opacity=0.1, style="surface", smooth_shading=True)
    
    print(f"Scaling neurons by {scale:.6f} and centering to match hand...")
    
    for i, n in enumerate(neurons):
        # Center and scale
        pts = n.points - neuron_center
        pts = pts * scale
        pts = pts + hand_center
        n.points = pts
        
        # Clip inside hand
        # PyVista 0.38+ clip_surface works well for solid geometries.
        try:
            print(f"Clipping neuron {i+1}/{len(neurons)}...")
            clipped = n.clip_surface(hand, invert=False)
            if clipped.n_points > 0:
                color = palette[i % len(palette)]
                plotter.add_mesh(clipped, color=color, smooth_shading=True, specular=0.5, ambient=0.2)
        except Exception as e:
            print(f"Error clipping neuron {i}: {e}")
            
    print("Exporting HTML...")
    plotter.export_html("interactive_hand.html")
    print("Saved interactive_hand.html")

if __name__ == "__main__":
    main()
