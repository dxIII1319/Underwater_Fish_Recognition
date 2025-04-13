# dataloader.py
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
from pathlib import Path
import numpy as np
from segmentation import load_segmentation_model
from saliency import SaliencyPartInitializer
# data_loader.py
class FishDataset(Dataset):
    def __init__(self, root, annotation_path=None, transforms=None, is_target=False):
        self.root = root
        self.transforms = transforms
        self.is_target = is_target  # <-- Add this flag
        
        # Paper 1 components
        self.saliency_init = SaliencyPartInitializer(num_parts=6)
        self.segmentor = load_segmentation_model()
        
        if annotation_path and not self.is_target:  # Source domain (labeled)
            self.coco = COCO(annotation_path)
            self.ids = list(sorted(self.coco.imgs.keys()))
        else:  # Target domain (unlabeled)
            self.ids = [f.stem for f in Path(root).glob("*.jpg")]

    def __getitem__(self, idx):
        # Load image
        if not self.is_target:  # Source domain (labeled)
            img_id = self.ids[idx]
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = f"{self.root}/images/{img_info['file_name']}"
        else:  # Target domain (unlabeled)
            img_path = f"{self.root}/{self.ids[idx]}.jpg"
        
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        
        # Segmentation (GrabCut) - Handle target domain differently
        if self.is_target:
            # For target domain, assume full image as foreground (no boxes)
            mask = self.segmentor.segment(img_np, boxes=None)
        else:
            # For source domain, use ground-truth boxes
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            boxes = [ann['bbox'] for ann in anns]
            mask = self.segmentor.segment(img_np, boxes)
        
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
        
        # Add annotations only for source domain
        if not self.is_target:
            target["boxes"] = torch.tensor([ann['bbox'] for ann in anns], dtype=torch.float32)
            target["labels"] = torch.tensor([ann['category_id'] for ann in anns], dtype=torch.int64)

        # Apply transforms
        if self.transforms:
            img = self.transforms(img)

        return img, target