"""
Extract frequency residuals from consecutive frames (Path Fixed)
"""
import numpy as np
import os
import json
import argparse
import multiprocessing as mp
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from frequency_tools import compute_fft_residual

class FrequencyExtractor:
    def __init__(self, root_dir: str, output_dir: str, method: str = 'fft'):
        # CHUẨN HÓA ĐƯỜNG DẪN NGAY TỪ ĐẦU
        self.root_dir = Path(root_dir).resolve() 
        self.output_dir = Path(output_dir).resolve()
        self.method = method
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_frame_pair(self, args):
        frame1_path, frame2_path = args
        try:
            # Load images
            img1 = np.array(Image.open(frame1_path).convert('RGB'))
            img2 = np.array(Image.open(frame2_path).convert('RGB'))
            
            # Compute residual
            residual = compute_fft_residual(img1, img2)
            
            # --- SỬA LỖI ĐƯỜNG DẪN ---
            frame1_path_obj = Path(frame1_path).resolve()
            
            try:
                # Cố gắng tìm đường dẫn tương đối chuẩn
                rel_path = frame1_path_obj.relative_to(self.root_dir)
            except ValueError:
                # Fallback cho Windows: Lấy tên folder cha (class name) thủ công
                # Ví dụ: D:/.../original/img.jpg -> class_name = original
                class_name = frame1_path_obj.parent.name
                rel_path = Path(class_name) / frame1_path_obj.name

            # Tạo folder output tương ứng
            save_dir = self.output_dir / rel_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = save_dir / f"{frame1_path_obj.stem}.npy"
            np.save(output_path, residual)
            return True
        except Exception as e:
            # Uncomment để debug nếu cần
            # print(f"Error processing {frame1_path}: {e}")
            return False
    
    def extract_from_splits(self, split_dir: str, num_workers: int = 8):
        split_files = list(Path(split_dir).glob("*.json"))
        all_pairs = []
        
        print(f"[Frequency] Scanning splits from {split_dir}...")
        
        for json_file in split_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            video_groups = defaultdict(list)
            for entry in data:
                # Xử lý đường dẫn trong JSON
                path_str = entry['path']
                # Thử tìm file
                if os.path.isabs(path_str) and os.path.exists(path_str):
                    abs_path = Path(path_str)
                else:
                    # Nối với root_dir project nếu là tương đối
                    # Giả sử script chạy từ root project
                    abs_path = Path(path_str).resolve()
                    
                if abs_path.exists():
                    filename = abs_path.name
                    # Lấy ID video (bỏ phần _f0.jpg)
                    if '_' in filename:
                        video_id = "_".join(filename.split('_')[:-1])
                    else:
                        video_id = abs_path.parent.name # Fallback
                    
                    video_groups[video_id].append(str(abs_path))
            
            for vid_id, frames in video_groups.items():
                frames.sort() # Đảm bảo thứ tự f0, f1, f2
                for i in range(len(frames) - 1):
                    all_pairs.append((frames[i], frames[i+1]))
        
        print(f"Total pairs found: {len(all_pairs)}")
        
        if num_workers > 1:
            with mp.Pool(num_workers) as pool:
                list(tqdm(pool.imap(self.process_frame_pair, all_pairs), total=len(all_pairs), desc="Extracting Frequency"))
        else:
            [self.process_frame_pair(p) for p in tqdm(all_pairs)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default="data/frames")
    parser.add_argument('--output_dir', type=str, default="data/frequency")
    parser.add_argument('--split_dir', type=str, default="splits")
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    
    extractor = FrequencyExtractor(args.root_dir, args.output_dir)
    extractor.extract_from_splits(args.split_dir, args.num_workers)

if __name__ == "__main__":
    main()