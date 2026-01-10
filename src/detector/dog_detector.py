"""
Dog detection and cropping using YOLOv8.
Based on ModelDevPart document - DETR or YOLO for dog region detection.
"""
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import torch

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")


class DogDetector:
    """
    Detects and crops dog regions from images using YOLOv8.
    Class 16 in COCO dataset = dog
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.25):
        """
        Initialize dog detector.
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detection
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package required. Install with: pip install ultralytics")
        
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.dog_class_id = 16  # COCO class 16 = dog
    
    def detect_dog(self, image_path: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect dog bounding box in image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Bounding box (x1, y1, x2, y2) or None if no dog detected
        """
        results = self.model(image_path, conf=self.conf_threshold, verbose=False)
        
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    if class_id == self.dog_class_id:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        return (x1, y1, x2, y2)
        
        return None
    
    def crop_dog(self, image_path: str, output_path: Optional[str] = None) -> Optional[Image.Image]:
        """
        Detect and crop dog region from image.
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save cropped image
            
        Returns:
            Cropped PIL Image or None if no dog detected
        """
        bbox = self.detect_dog(image_path)
        
        if bbox is None:
            return None
        
        x1, y1, x2, y2 = bbox
        
        # Load and crop image
        img = Image.open(image_path).convert('RGB')
        cropped = img.crop((x1, y1, x2, y2))
        
        if output_path:
            cropped.save(output_path)
        
        return cropped
    
    def detect_and_crop_batch(self, image_paths: list) -> list:
        """
        Process multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of cropped PIL Images (None for failed detections)
        """
        results = []
        for img_path in image_paths:
            cropped = self.crop_dog(img_path)
            results.append(cropped)
        return results


def detect_dog_simple(image_path: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Simple wrapper function for quick dog detection.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Bounding box (x1, y1, x2, y2) or None
    """
    detector = DogDetector()
    return detector.detect_dog(image_path)

