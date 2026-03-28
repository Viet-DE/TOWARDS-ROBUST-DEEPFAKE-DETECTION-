"""
TriXNet Inference Pipeline (End-to-End)
Chạy dự đoán trực tiếp trên Video hoặc Webcam.
Tự động Preprocessing on-the-fly (không cần tạo file .npy).
"""
import os
import cv2
import torch
import argparse
import numpy as np
import mediapipe as mp
from PIL import Image
import torchvision.transforms as T
import time

# Import Tools (Tái sử dụng các hàm toán học)
from preprocessing.frequency.frequency_tools import compute_fft_residual
from preprocessing.flow.optical_flow import compute_farneback_flow, normalize_flow
from preprocessing.parts.parts_utils import extract_facial_parts

# Import Model
from models.trixnet import create_model
from utils import get_config

class TriXNetPredictor:
    def __init__(self, config_path, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 1. Load Model
        print(f"Loading TriXNet on {self.device}...")
        self.config = get_config(config_path)
        self.model = create_model(self.config).to(self.device)
        self.model.eval()
        
        # Load weights
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['state_dict'])
            print("Model weights loaded successfully.")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # 2. Setup Face Detector (MediaPipe)
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        
        # 3. Setup Transform (Chuẩn hóa giống lúc train)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Buffer lưu frame trước để tính Flow/Residual
        self.prev_frame_crop = None

    def crop_face(self, frame):
        """Cắt mặt từ frame (Logic giống prepare_ffpp.py)"""
        h_img, w_img, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(img_rgb)
        
        if not results.detections:
            return None, None
            
        # Lấy mặt to nhất
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        
        x = int(bboxC.xmin * w_img)
        y = int(bboxC.ymin * h_img)
        w = int(bboxC.width * w_img)
        h = int(bboxC.height * h_img)
        
        # Scale 1.3x
        scale = 1.3
        center_x, center_y = x + w//2, y + h//2
        crop_size = int(max(w, h) * scale)
        
        x1 = max(0, center_x - crop_size//2)
        y1 = max(0, center_y - crop_size//2)
        x2 = min(w_img, center_x + crop_size//2)
        y2 = min(h_img, center_y + crop_size//2)
        
        face = frame[y1:y2, x1:x2]
        if face.size == 0: return None, None
        
        # Resize về chuẩn 256x256 để tính toán feature
        face_resized = cv2.resize(face, (256, 256))
        
        # Trả về cả box để vẽ visualization
        return face_resized, (x1, y1, x2, y2)

    def preprocess_and_predict(self, curr_frame_crop):
        """
        Xử lý 1 frame đã crop và đưa vào model
        Cần frame hiện tại (t) và frame trước đó (t-1)
        """
        if self.prev_frame_crop is None:
            self.prev_frame_crop = curr_frame_crop
            return None # Cần ít nhất 2 frame để bắt đầu
            
        # --- 1. Prepare Inputs ---
        img1 = self.prev_frame_crop # t-1
        img2 = curr_frame_crop      # t
        
        # A. Frequency (FFT Residual)
        # Input RGB -> Output Grayscale [H, W]
        freq_map = compute_fft_residual(img1, img2)
        freq_tensor = torch.from_numpy(freq_map).unsqueeze(0).float() # [1, H, W]
        # Resize về 224 (Model Input)
        freq_tensor = torch.nn.functional.interpolate(freq_tensor.unsqueeze(0), size=(224, 224)).squeeze(0)
        
        # B. Optical Flow
        # Input RGB -> Output [H, W, 2]
        flow_map = compute_farneback_flow(img1, img2)
        flow_norm = normalize_flow(flow_map)
        flow_tensor = torch.from_numpy(flow_norm).permute(2, 0, 1).float() # [2, H, W]
        flow_tensor = torch.nn.functional.interpolate(flow_tensor.unsqueeze(0), size=(224, 224)).squeeze(0)
        
        # C. Parts (Eyes/Mouth)
        # Input RGB -> Dict
        parts_data = extract_facial_parts(img2) # Dùng frame hiện tại
        if parts_data is None:
            # Fallback nếu MediaPipe Landmarks thất bại
            eyes = torch.zeros(3, 64, 128)
            mouth = torch.zeros(3, 64, 64)
        else:
            eyes = torch.from_numpy(parts_data['eyes']).permute(2, 0, 1).float() / 255.0
            mouth = torch.from_numpy(parts_data['mouth']).permute(2, 0, 1).float() / 255.0
            
        # D. RGB Image (Optional nếu model cần)
        # img_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        # img_tensor = self.transform(img_pil)

        # --- 2. Batching ---
        inputs = {
            'frequency': freq_tensor.unsqueeze(0).to(self.device), # [1, 1, 224, 224]
            'flow': flow_tensor.unsqueeze(0).to(self.device),      # [1, 2, 224, 224]
            'parts': {
                'eyes': eyes.unsqueeze(0).to(self.device),
                'mouth': mouth.unsqueeze(0).to(self.device)
            }
        }
        
        # --- 3. Inference ---
        with torch.no_grad():
            logits, _, _ = self.model(inputs)
            probs = torch.softmax(logits, dim=1)
            fake_prob = probs[0, 1].item()
            
        # Update buffer
        self.prev_frame_crop = curr_frame_crop
        
        return fake_prob

def run_inference(source, config, checkpoint):
    predictor = TriXNetPredictor(config, checkpoint)
    
    # Mở Video hoặc Webcam
    if source.isdigit():
        cap = cv2.VideoCapture(int(source)) # Webcam
    else:
        cap = cv2.VideoCapture(source)      # File path
        
    if not cap.isOpened():
        print(f"Cannot open source: {source}")
        return

    print("Starting Inference... Press 'q' to quit.")
    
    frame_count = 0
    fps_time = time.time()
    
    # Biến làm mượt kết quả (Moving Average)
    prob_buffer = []
    SMOOTHING_WINDOW = 5
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Resize cho nhẹ nếu video 4K
        if frame.shape[1] > 1280:
            frame = cv2.resize(frame, (1280, 720))
            
        # 2. Detect & Crop
        face_crop, bbox = predictor.crop_face(frame)
        
        fake_prob = 0.0
        label = "Processing..."
        color = (255, 255, 0) # Cyan (Neutral)
        
        if face_crop is not None:
            # 3. Predict
            raw_prob = predictor.preprocess_and_predict(face_crop)
            
            if raw_prob is not None:
                # Smoothing
                prob_buffer.append(raw_prob)
                if len(prob_buffer) > SMOOTHING_WINDOW:
                    prob_buffer.pop(0)
                
                avg_prob = sum(prob_buffer) / len(prob_buffer)
                
                # Decision logic
                if avg_prob > 0.7:
                    label = f"FAKE ({avg_prob*100:.1f}%)"
                    color = (0, 0, 255) # RED
                else:
                    label = f"REAL ({(1-avg_prob)*100:.1f}%)"
                    color = (0, 255, 0) # GREEN
                
                # Vẽ Box
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Vẽ Label nền đen cho dễ đọc
                cv2.rectangle(frame, (x1, y1-30), (x2, y1), color, -1)
                cv2.putText(frame, label, (x1+5, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # FPS calculation
        frame_count += 1
        if frame_count % 10 == 0:
            curr_time = time.time()
            fps = 10 / (curr_time - fps_time)
            fps_time = curr_time
            print(f"\rFPS: {fps:.1f} | {label}", end="")
            
        cv2.imshow('TriXNet Deepfake Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='Path to video file or 0 for webcam')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/trixnet/model_best.pth')
    
    args = parser.parse_args()
    
    run_inference(args.source, args.config, args.checkpoint)