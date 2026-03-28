"""
Master script để chạy toàn bộ quy trình chuẩn bị dữ liệu.
Chạy file này sẽ kích hoạt lần lượt:
1. Crop Face & Split Data
2. Extract Frequency
3. Extract Flow
4. Extract Parts
"""
import os
import sys
import subprocess
import time

# Định nghĩa Python executable (để đảm bảo dùng đúng venv hiện tại)
PYTHON_EXE = sys.executable

def run_command(cmd, step_name):
    print(f"\n{'='*60}")
    print(f" STARTING: {step_name}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        start_time = time.time()
        subprocess.check_call(cmd)
        elapsed = time.time() - start_time
        print(f"\n FINISHED: {step_name} in {elapsed:.2f}s")
    except subprocess.CalledProcessError as e:
        print(f"\n ERROR in {step_name}. Exit code: {e.returncode}")
        sys.exit(1)

def main():
    # 1. Prepare Data (Crop Faces & Create JSON Splits)
    # Lệnh này gọi file scripts/prepare_ffpp.py
    run_command([PYTHON_EXE, "scripts/prepare_ffpp.py"], 
                "STEP 1: Face Extraction & Splitting")

    # Các tham số chung cho bước extract feature
    # Lưu ý: Các file extract nằm trong preprocessing/..., không phải scripts/
    NUM_WORKERS = "4" # Giảm xuống nếu máy lag
    
    # 2. Extract Frequency
    run_command([PYTHON_EXE, "preprocessing/frequency/extract_frequency.py", 
                 "--num_workers", NUM_WORKERS,
                 "--split_dir", "splits"], 
                "STEP 2: Frequency Extraction (FFT)")

    # 3. Extract Optical Flow
    run_command([PYTHON_EXE, "preprocessing/flow/extract_flow.py", 
                 "--num_workers", NUM_WORKERS,
                 "--split_dir", "splits"], 
                "STEP 3: Optical Flow Extraction")

    # 4. Extract Parts (Eyes/Mouth)
    run_command([PYTHON_EXE, "preprocessing/parts/extract_parts.py", 
                 "--num_workers", NUM_WORKERS,
                 "--split_dir", "splits"], 
                "STEP 4: Facial Parts Extraction")

    print(f"\n{'='*60}")
    print("ALL PREPROCESSING STEPS COMPLETED SUCCESSFULLY!")
    print("Next step: python train.py")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()