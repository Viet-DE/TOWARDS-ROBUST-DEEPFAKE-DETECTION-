"""
Frequency domain tools for FRS branch
"""
import numpy as np
import cv2
from typing import Tuple


def compute_fft_residual(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Compute FFT residual between two frames
    
    Args:
        img1, img2: RGB images [H, W, 3]
        
    Returns:
        residual: [H, W] frequency residual map
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Apply 2D FFT
    fft1 = np.fft.fft2(gray1)
    fft2 = np.fft.fft2(gray2)
    
    # Shift zero frequency to center
    fft1_shift = np.fft.fftshift(fft1)
    fft2_shift = np.fft.fftshift(fft2)
    
    # Calculate amplitude spectrum
    amplitude1 = np.abs(fft1_shift)
    amplitude2 = np.abs(fft2_shift)
    
    # Residual
    residual = np.abs(amplitude1 - amplitude2)
    
    # Log scale for better visualization
    residual = np.log(residual + 1)
    
    # Normalize to [0, 1]
    residual = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
    
    return residual.astype(np.float32)


def compute_dct_residual(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Compute DCT residual between two frames
    
    Args:
        img1, img2: RGB images [H, W, 3]
        
    Returns:
        residual: [H, W] DCT residual map
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Apply DCT
    dct1 = cv2.dct(gray1)
    dct2 = cv2.dct(gray2)
    
    # Residual
    residual = np.abs(dct1 - dct2)
    
    # Log scale
    residual = np.log(residual + 1)
    
    # Normalize
    residual = (residual - residual.min()) / (residual.max() - residual.min() + 1e-8)
    
    return residual.astype(np.float32)


def visualize_frequency_spectrum(fft_shift: np.ndarray) -> np.ndarray:
    """
    Visualize frequency spectrum
    
    Args:
        fft_shift: Shifted FFT result
        
    Returns:
        vis: [H, W, 3] RGB visualization
    """
    magnitude = np.abs(fft_shift)
    magnitude = np.log(magnitude + 1)
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    magnitude = (magnitude * 255).astype(np.uint8)
    
    # Convert to RGB
    vis = cv2.applyColorMap(magnitude, cv2.COLORMAP_JET)
    
    return vis