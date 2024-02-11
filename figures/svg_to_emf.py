import os
import subprocess

# Get the current directory
current_dir = os.path.dirname(os.path.realpath(__file__))

# Loop over all files in the current directory
for filename in os.listdir(current_dir):
    # If the file is an SVG file
    if filename.endswith('.svg'):
        # Get the base name of the file (without the .svg extension)
        base_filename = os.path.splitext(filename)[0]
        # Convert the SVG file to an EMF file using Inkscape
        subprocess.run(['inkscape', filename, '--export-type=emf', '--export-filename=' + f"{base_filename}.emf"])