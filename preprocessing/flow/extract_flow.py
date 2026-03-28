
import numpy as np
import json
import argparse
import multiprocessing as mp
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from optical_flow import compute_farneback_flow, normalize_flow
import os

class OpticalFlowExtractor:
    def __init__(self, root_dir: str, output_dir: str, method: str = 'farneback'):
        self.root_dir = Path(root_dir).resolve() # Resolve ngay
        self.output_dir = Path(output_dir).resolve()
        self.method = method
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_frame_pair(self, args):
        frame1_path, frame2_path = args
        try:
            img1 = np.array(Image.open(frame1_path).convert('RGB'))
            img2 = np.array(Image.open(frame2_path).convert('RGB'))
            
            flow = compute_farneback_flow(img1, img2)
            flow_norm = normalize_flow(flow)
            
            # --- FIX PATH ---
            frame1_path_obj = Path(frame1_path).resolve()
            try:
                rel_path = frame1_path_obj.relative_to(self.root_dir)
            except ValueError:
                class_name = frame1_path_obj.parent.name
                rel_path = Path(class_name) / frame1_path_obj.name

            save_dir = self.output_dir / rel_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = save_dir / f"{frame1_path_obj.stem}.npy"
            np.save(output_path, flow_norm)
            return True
        except Exception:
            return False
    
    def extract_from_splits(self, split_dir: str, num_workers: int = 8):
        split_files = list(Path(split_dir).glob("*.json"))
        all_pairs = []
        print(f"[Flow] Scanning splits from {split_dir}...")
        
        for json_file in split_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            video_groups = defaultdict(list)
            for entry in data:
                path_str = entry['path']
                if os.path.isabs(path_str) and os.path.exists(path_str):
                    abs_path = Path(path_str)
                else:
                    abs_path = Path(path_str).resolve()
                
                if abs_path.exists():
                    filename = abs_path.name
                    if '_' in filename:
                        video_id = "_".join(filename.split('_')[:-1])
                    else:
                        video_id = abs_path.parent.name
                    video_groups[video_id].append(str(abs_path))
            
            for vid_id, frames in video_groups.items():
                frames.sort()
                for i in range(len(frames) - 1):
                    all_pairs.append((frames[i], frames[i+1]))
                    
        print(f"Total pairs found: {len(all_pairs)}")
        if num_workers > 1:
            with mp.Pool(num_workers) as pool:
                list(tqdm(pool.imap(self.process_frame_pair, all_pairs), total=len(all_pairs), desc="Extracting Flow"))
        else:
            [self.process_frame_pair(p) for p in tqdm(all_pairs)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="data/frames")
    parser.add_argument('--output_dir', type=str, default="data/flow")
    parser.add_argument('--split_dir', type=str, default="splits")
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    extractor = OpticalFlowExtractor(args.root_dir, args.output_dir)
    extractor.extract_from_splits(args.split_dir, args.num_workers)

if __name__ == "__main__":
    main()