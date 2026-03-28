"""
Extract facial parts (Path Fixed)
"""
import numpy as np
import json
import argparse
import multiprocessing as mp
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from parts_utils import extract_facial_parts
import os

class LocalPartsExtractor:
    def __init__(self, root_dir: str, output_dir: str):
        self.root_dir = Path(root_dir).resolve() # Resolve ngay
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_frame(self, args):
        frame_path, save_vis = args
        try:
            image = np.array(Image.open(frame_path).convert('RGB'))
            parts = extract_facial_parts(image)
            if parts is None: return False
            
            # --- FIX PATH ---
            frame_path_obj = Path(frame_path).resolve()
            try:
                rel_path = frame_path_obj.relative_to(self.root_dir)
            except ValueError:
                class_name = frame_path_obj.parent.name
                rel_path = Path(class_name) / frame_path_obj.name
            
            save_dir = self.output_dir / rel_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = save_dir / f"{frame_path_obj.stem}.npz"
            np.savez_compressed(
                output_path,
                eyes=parts['eyes'],
                mouth=parts['mouth']
            )
            return True
        except Exception:
            return False
    
    def extract_from_splits(self, split_dir: str, num_workers: int = 8, save_vis: bool = False):
        split_files = list(Path(split_dir).glob("*.json"))
        all_frames = []
        print(f"[Parts] Scanning splits from {split_dir}...")
        
        for json_file in split_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            for entry in data:
                path_str = entry['path']
                if os.path.isabs(path_str) and os.path.exists(path_str):
                    abs_path = Path(path_str)
                else:
                    abs_path = Path(path_str).resolve()
                
                if abs_path.exists():
                    all_frames.append((str(abs_path), save_vis))
        
        print(f"Total frames: {len(all_frames)}")
        if num_workers > 1:
            with mp.Pool(num_workers) as pool:
                list(tqdm(pool.imap(self.process_frame, all_frames), total=len(all_frames), desc="Extracting Parts"))
        else:
            [self.process_frame(f) for f in tqdm(all_frames)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="data/frames")
    parser.add_argument('--output_dir', type=str, default="data/parts")
    parser.add_argument('--split_dir', type=str, default="splits")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_vis', action='store_true')
    args = parser.parse_args()
    
    extractor = LocalPartsExtractor(args.root_dir, args.output_dir)
    extractor.extract_from_splits(args.split_dir, args.num_workers, args.save_vis)

if __name__ == "__main__":
    main()