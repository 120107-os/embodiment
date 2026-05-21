import pyvista as pv
import numpy as np

plotter = pv.Plotter(off_screen=True)

# Add a cube
cube = pv.Cube()
plotter.add_mesh(cube, color='red')

# Add lines
lines = pv.BoxBounds(cube.bounds)
plotter.add_mesh(lines, color='white', style='wireframe')

plotter.export_gltf('test_scene.gltf')
print("GLTF exported successfully.")
