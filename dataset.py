import torch
from PIL import Image, ImageOps 
from args import get_args
from utils import resize_box_xyxy
import augmentations as aug

class ObjDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = aug.Compose(transform) if transform else aug.Compose([aug.ToTensor()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        args = get_args()
        row = self.df.iloc[idx]

        img = Image.open(row["image_path"]).convert("RGB")
        img = ImageOps.exif_transpose(img)  
        w, h = img.size

        boxes, labels = [], []
        with open(row["label_path"]) as f:
            for line in f:
                cls, xc, yc, bw, bh = map(float, line.split())

                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w
                y2 = (yc + bh / 2) * h
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls) + 1)

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
        }

        image, target = self.transform(img, target)  
        return image, target