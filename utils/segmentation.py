# segmentation.py
import cv2
import numpy as np

class GrabCutSegmentor:
    """
    Implements Paper 1's segmentation method using GrabCut initialized with bounding boxes.
    """
    
    def __init__(self, num_iter=5):
        self.num_iter = num_iter  # Number of GrabCut iterations (Paper 1 uses 5)

    def segment(self, image, boxes=None):
        """
        Segment fish using GrabCut initialized with bounding boxes.
        Args:
            image: Input image (numpy array, HxWx3)
            boxes: List of bounding boxes [[x,y,w,h], ...] (required for source domain)
        Returns:
            mask: Segmentation mask (HxW), 1=foreground, 0=background
        """
        if boxes is None:  # For target domain (no annotations)
            return self._segment_without_boxes(image)
        else:
            return self._segment_with_boxes(image, boxes)

    def _segment_with_boxes(self, image, boxes):
        # Initialize mask with probable background
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Set regions from bounding boxes as probable foreground
        for box in boxes:
            x, y, w, h = map(int, box)
            mask[y:y+h, x:x+w] = cv2.GC_PR_FGD
        
        # Run GrabCut (Paper 1 uses 5 iterations)
        mask, _, _ = cv2.grabCut(
            image, mask, None, bgd_model, fgd_model,
            iterCount=self.num_iter, mode=cv2.GC_INIT_WITH_MASK
        )
        
        # Convert mask to binary
        return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.float32)

    def _segment_without_boxes(self, image):
        """Fallback for target domain (full-image foreground assumption)"""
        return np.ones(image.shape[:2], dtype=np.float32)

def load_segmentation_model():
    return GrabCutSegmentor()