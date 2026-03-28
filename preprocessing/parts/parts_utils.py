"""
Utility functions for facial parts extraction using MediaPipe
"""
import numpy as np
import cv2
import mediapipe as mp


# MediaPipe Face Mesh landmark indices
LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]


def get_face_mesh():
    """Initialize MediaPipe Face Mesh"""
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


def get_landmarks(image: np.ndarray, face_mesh):
    """
    Detect facial landmarks using MediaPipe
    
    Args:
        image: RGB image [H, W, 3]
        face_mesh: MediaPipe FaceMesh instance
        
    Returns:
        landmarks: [468, 2] numpy array or None
    """
    results = face_mesh.process(image)
    
    if not results.multi_face_landmarks:
        return None
    
    # Get first face
    face_landmarks = results.multi_face_landmarks[0]
    
    # Convert to numpy array
    h, w = image.shape[:2]
    landmarks = np.array([
        [lm.x * w, lm.y * h] 
        for lm in face_landmarks.landmark
    ])
    
    return landmarks


def crop_region(image: np.ndarray, 
                landmarks: np.ndarray, 
                indices: list, 
                padding: float = 0.2) -> np.ndarray:
    """
    Crop a facial region based on landmarks
    
    Args:
        image: RGB image [H, W, 3]
        landmarks: [468, 2] landmark coordinates
        indices: landmark indices for the region
        padding: padding percentage around bounding box
        
    Returns:
        cropped: cropped region
    """
    points = landmarks[indices]
    
    # Calculate bounding box
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    
    # Add padding
    w = x_max - x_min
    h = y_max - y_min
    
    x_min = int(max(0, x_min - w * padding))
    y_min = int(max(0, y_min - h * padding))
    x_max = int(min(image.shape[1], x_max + w * padding))
    y_max = int(min(image.shape[0], y_max + h * padding))
    
    # Crop
    cropped = image[y_min:y_max, x_min:x_max]
    
    return cropped


def extract_facial_parts(image: np.ndarray, 
                         target_size: tuple = (64, 64)):
    """
    Extract eyes and mouth from image
    
    Args:
        image: RGB image [H, W, 3]
        target_size: resize target for parts
        
    Returns:
        dict with 'eyes', 'mouth', 'landmarks' or None
    """
    face_mesh = get_face_mesh()
    landmarks = get_landmarks(image, face_mesh)
    face_mesh.close()
    
    if landmarks is None:
        return None
    
    # Crop eyes
    left_eye = crop_region(image, landmarks, LEFT_EYE_INDICES)
    right_eye = crop_region(image, landmarks, RIGHT_EYE_INDICES)
    
    # Resize and combine eyes
    left_eye = cv2.resize(left_eye, target_size)
    right_eye = cv2.resize(right_eye, target_size)
    eyes_combined = np.concatenate([left_eye, right_eye], axis=1)  # [64, 128, 3]
    
    # Crop and resize mouth
    mouth = crop_region(image, landmarks, MOUTH_INDICES)
    mouth = cv2.resize(mouth, target_size)  # [64, 64, 3]
    
    return {
        'eyes': eyes_combined,
        'mouth': mouth,
        'landmarks': landmarks
    }


def visualize_landmarks(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Draw landmarks on image for visualization"""
    vis = image.copy()
    
    # Draw all landmarks
    for x, y in landmarks:
        cv2.circle(vis, (int(x), int(y)), 1, (0, 255, 0), -1)
    
    # Draw bounding boxes
    for indices, color, name in [
        (LEFT_EYE_INDICES, (255, 0, 0), 'Left Eye'),
        (RIGHT_EYE_INDICES, (255, 0, 0), 'Right Eye'),
        (MOUTH_INDICES, (0, 0, 255), 'Mouth')
    ]:
        points = landmarks[indices]
        x_min, y_min = points.min(axis=0).astype(int)
        x_max, y_max = points.max(axis=0).astype(int)
        cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(vis, name, (x_min, y_min - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return vis