
import numpy as np
import cv2


def compute_farneback_flow(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Compute optical flow using Farneback method
    
    Args:
        img1, img2: RGB images [H, W, 3]
        
    Returns:
        flow: [H, W, 2] optical flow (dx, dy)
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    # Calculate flow
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    
    return flow.astype(np.float32)


def normalize_flow(flow: np.ndarray) -> np.ndarray:
    """Normalize flow to [-1, 1]"""
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    max_mag = np.percentile(mag, 99)
    flow_norm = flow / (max_mag + 1e-8)
    flow_norm = np.clip(flow_norm, -1, 1)
    return flow_norm.astype(np.float32)


def visualize_flow(flow: np.ndarray) -> np.ndarray:
    """Visualize optical flow as HSV image"""
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb