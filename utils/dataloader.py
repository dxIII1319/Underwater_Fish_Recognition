# dataloader.py
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
from pathlib import Path
import numpy as np
from segmentation import load_segmentation_model
from saliency import SaliencyPartInitializer

class FishDataset(Dataset):
    def __init__(self, root, annotation_path=None, transforms=None, is_target=False):
        self.root = root
        self.transforms = transforms
        self.is_target = is_target
        
        # Paper 1 components
        self.saliency_init = SaliencyPartInitializer(num_parts=6)
        self.segmentor = load_segmentation_model()
        
        if annotation_path:  # Source domain (labeled)
            self.coco = COCO(annotation_path)
            self.ids = list(sorted(self.coco.imgs.keys()))
        else:  # Target domain (unlabeled)
            self.ids = [f.stem for f in Path(root).glob("*.jpg")]

    def __getitem__(self, idx):
        # Load image
        if hasattr(self, 'coco'):  # Source domain
            img_id = self.ids[idx]
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = f"{self.root}/images/{img_info['file_name']}"
        else:  # Target domain
            img_path = f"{self.root}/{self.ids[idx]}.jpg"
            
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)  # Convert to numpy for segmentation
        
        # Get bounding boxes (source domain only)
        boxes = []
        if hasattr(self, 'coco'):
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            boxes = [ann['bbox'] for ann in anns]
        
        # Segment using Paper 1's GrabCut method
        mask = self.segmentor.segment(img_np, boxes if not self.is_target else None)
        
        # Saliency-based part initialization
        saliency_points = self.saliency_init(img_np, mask)
        H, W = img_np.shape[:2]
        saliency_points = torch.tensor(saliency_points, dtype=torch.float32)
        saliency_points[:, 0] /= H  # Normalize y
        saliency_points[:, 1] /= W  # Normalize x
        
        # Prepare target
        target = {
            "saliency_points": saliency_points,
            "mask": torch.from_numpy(mask).float()
        }
        
        # Add annotations for source domain
        if hasattr(self, 'coco'):
            target["boxes"] = torch.tensor([ann['bbox'] for ann in anns], dtype=torch.float32)
            target["labels"] = torch.tensor([ann['category_id'] for ann in anns], dtype=torch.int64)

        # Apply transforms
        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)