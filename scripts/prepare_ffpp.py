import os
import cv2
import json
import mediapipe as mp
from tqdm import tqdm
from pathlib import Path

# --- CẤU HÌNH ---
# Đường dẫn tới dataset gốc
SOURCE_PATH = r"D:\Download Anything\School\ffpp extracted datasets\FF++C32-Frames"

# Đường dẫn project
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

OUTPUT_FRAMES_DIR = os.path.join(PROJECT_ROOT, "data", "frames")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "splits")

# Config
IMG_SIZE = 256
SCALE = 1.3 

# MediaPipe
mp_face_detection = mp.solutions.face_detection
detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def get_split(video_id):
    try:
        vid_id_int = int(video_id)
        if vid_id_int < 720: return "train"
        elif vid_id_int < 860: return "val"
        else: return "test"
    except:
        return "train"

def setup_dirs():
    os.makedirs(SPLITS_DIR, exist_ok=True)
    # Xóa dữ liệu cũ nếu cần thiết hoặc chỉ overwrite
    classes = ["Original", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT_FRAMES_DIR, cls.lower()), exist_ok=True)

def find_images_recursive(root_dir):
    """Tìm tất cả ảnh jpg/png trong folder và các folder con"""
    image_files = []
    print(f"Scanning: {root_dir}...")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_files.append(os.path.join(root, file))
    return image_files

def crop_face_mediapipe(image):
    if image is None: return None
    h_img, w_img, _ = image.shape
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.process(img_rgb)
    
    if not results.detections: return None
    
    detection = results.detections[0]
    bboxC = detection.location_data.relative_bounding_box
    
    x = int(bboxC.xmin * w_img)
    y = int(bboxC.ymin * h_img)
    w = int(bboxC.width * w_img)
    h = int(bboxC.height * h_img)
    
    center_x, center_y = x + w // 2, y + h // 2
    crop_size = int(max(w, h) * SCALE)
    
    x1 = max(0, center_x - crop_size // 2)
    y1 = max(0, center_y - crop_size // 2)
    x2 = min(w_img, center_x + crop_size // 2)
    y2 = min(h_img, center_y + crop_size // 2)
    
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0: return None
    return cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))

def main():
    print(f"[Step 1] RECURSIVE SCAN MODE")
    setup_dirs()
    
    data_splits = {"train": [], "val": [], "test": []}
    
    folder_map = {
        "Original": "original",
        "Deepfakes": "deepfakes",
        "Face2Face": "face2face", 
        "FaceShifter": "faceshifter",
        "FaceSwap": "faceswap",
        "NeuralTextures": "neuraltextures"
    }

    total_processed = 0
    
    for src_folder, target_label in folder_map.items():
        src_dir_full = os.path.join(SOURCE_PATH, src_folder)
        
        # Kiểm tra folder nguồn
        if not os.path.exists(src_dir_full):
            print(f"CẢNH BÁO: Không tìm thấy folder nguồn: {src_dir_full}")
            continue
            
        # Tìm ảnh đệ quy (Recursive)
        all_images = find_images_recursive(src_dir_full)
        print(f"Tìm thấy {len(all_images)} ảnh trong nhóm {src_folder}")
        
        if len(all_images) == 0:
            print(f"LỖI: Folder {src_folder} rỗng! Hãy kiểm tra lại ổ D.")
            continue

        for img_path in tqdm(all_images, desc=f"Processing {target_label}"):
            fname = os.path.basename(img_path)
            
            # Logic lấy ID video từ tên file hoặc tên folder cha
            # FF++ thường là: 000_f0.jpg HOẶC 000/frame.jpg
            try:
                if '_' in fname:
                    video_id = fname.split('_')[0]
                else:
                    # Nếu file nằm trong folder con (ví dụ 000/frame.jpg), lấy tên folder cha làm ID
                    parent_name = os.path.basename(os.path.dirname(img_path))
                    if parent_name.isdigit():
                        video_id = parent_name
                    else:
                        continue # Bỏ qua nếu không xác định được ID
                
                split_mode = get_split(video_id)
            except: continue

            image = cv2.imread(img_path)
            face_img = crop_face_mediapipe(image)
            
            if face_img is not None:
                # Đặt tên file mới để tránh trùng lặp
                # new_name = videoID_filename
                save_name = f"{video_id}_{fname}"
                save_path = os.path.join(OUTPUT_FRAMES_DIR, target_label, save_name)
                
                cv2.imwrite(save_path, face_img)
                
                rel_path = os.path.join("data", "frames", target_label, save_name).replace("\\", "/")
                
                data_splits[split_mode].append({
                    "path": rel_path,
                    "label": 0 if target_label == "original" else 1,
                    "class_name": target_label
                })
                total_processed += 1

    print("Saving split JSONs...")
    for mode, data in data_splits.items():
        json_path = os.path.join(SPLITS_DIR, f"{mode}.json")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"   -> Saved {mode}.json: {len(data)} items")
            
    print(f"Step 1 Done. Processed {total_processed} frames.")

if __name__ == "__main__":
    main()