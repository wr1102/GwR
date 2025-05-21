import numpy as np
import os
import argparse
from tqdm import tqdm

def add_noise_and_save(pdb_lines, variance, output_file):
    with open(output_file, "w") as file:
        for line in pdb_lines:
            if line.startswith("ATOM"):
                parts = line.split()
                try:
                    coords = np.array([float(parts[6]), float(parts[7]), float(parts[8])])
                    noise = np.random.normal(0, variance, coords.shape)
                    new_coords = coords + noise
                    new_line = f"{line[:30]}{new_coords[0]:8.3f}{new_coords[1]:8.3f}{new_coords[2]:8.3f}{line[54:]}"
                    file.write(new_line + "\n")
                except Exception:
                    file.write(line + "\n")
            else:
                file.write(line + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--variance', type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pdb_files = [f for f in os.listdir(args.input_dir) if f.endswith('.pdb')]
    
    for pdb in tqdm(pdb_files):
        input_path = os.path.join(args.input_dir, pdb)
        output_path = os.path.join(args.output_dir, pdb)

        with open(input_path, 'r') as f:
            pdb_lines = f.read().splitlines()

        try:
            add_noise_and_save(pdb_lines, args.variance, output_path)
        except Exception as e:
            print(f"Error processing {pdb}: {e}")
