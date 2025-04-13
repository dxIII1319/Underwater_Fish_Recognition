# saliency.py
import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

class SaliencyPartInitializer:
    """
    Implements Paper 1's saliency-based part initialization using PFT (Phase Fourier Transform).
    Used to initialize object parts before training the non-rigid part model.
    """
    
    def __init__(self, num_parts=6, saliency_threshold=0.8, sigma=3.0):
        self.num_parts = num_parts          # K in Paper 1 (e.g., 6 parts)
        self.saliency_threshold = saliency_threshold  # Threshold for salient regions
        self.sigma = sigma                  # Gaussian blur parameter

    def compute_pft_saliency(self, image):
        """
        Compute PFT saliency map (Eq. 9-10 in Paper 1)
        Args:
            image: Input image (numpy array, HxWx3)
        Returns:
            saliency_map: Saliency heatmap (HxW)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Compute 2D FFT
        fft = np.fft.fft2(gray)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        
        # Inverse FFT using only phase (Eq. 10)
        saliency = np.fft.ifft2(np.exp(1j * phase)).real
        
        # Post-processing
        saliency = np.abs(saliency) ** 2
        saliency = gaussian_filter(saliency, sigma=self.sigma)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        return saliency

    def detect_salient_points(self, saliency_map, mask=None):
        """
        Detect top-K salient points (Algorithm 1 in Paper 1)
        Args:
            saliency_map: Output from compute_pft_saliency()
            mask: Segmentation mask (HxW) to filter background (from GrabCut/UNet)
        Returns:
            points: List of (y, x) coordinates of salient points
        """
        # Thresholding
        saliency_map[saliency_map < self.saliency_threshold] = 0
        
        # Non-maximal suppression
        coordinates = peak_local_max(
            saliency_map, 
            min_distance=10, 
            num_peaks=self.num_parts,
            exclude_border=False
        )
        
        # Filter using segmentation mask
        if mask is not None:
            valid_points = []
            for y, x in coordinates:
                if mask[y, x] > 0:  # Check if point is within object mask
                    valid_points.append((y, x))
            coordinates = np.array(valid_points)
        
        return coordinates[:self.num_parts]  # Return top-K points

    def __call__(self, image, mask=None):
        """ 
        Full pipeline: Compute saliency → detect points → return coordinates 
        """
        saliency = self.compute_pft_saliency(image)
        points = self.detect_salient_points(saliency, mask)
        return points  # Shape: (K, 2)


def visualize_saliency(image, points):
    """ 
    Helper function to visualize saliency points 
    (For debugging/paper figures)
    """
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.scatter(points[:,1], points[:,0], c='r', s=50)
    plt.title('Saliency-based Part Initialization')
    plt.show()