import torch
from PIL import Image, ImageOps, ImageDraw
from torchvision.transforms.functional import to_tensor
import os

from model import build_model
from args import get_args

def predict_images(image_folder: str, model_path: str, score_threshold: float = 0.5):
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model 
    model = build_model(args.backbone, num_classes=args.num_classes + 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded model from {model_path}")

    # Create a folder to save test results
    out_folder = "./test_results"
    os.makedirs(out_folder, exist_ok=True)

    image_files = [f for f in os.listdir(image_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f" Found {len(image_files)} images")

    for filename in image_files:
        img_path = os.path.join(image_folder, filename)

        # Open test images and trapose EXIF
        img = Image.open(img_path).convert("RGB")
        img = ImageOps.exif_transpose(img)
        orig_w, orig_h = img.size

        # Resize and tensor test images
        resized = img.resize((args.image_size, args.image_size))
        tensor  = to_tensor(resized).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            predictions = model(tensor)

        boxes  = predictions[0]["boxes"]
        scores = predictions[0]["scores"]

        # filter box with high scores
        keep   = scores > score_threshold
        boxes  = boxes[keep].cpu()
        scores = scores[keep].cpu()

        # Scale box into original size
        scale_x = orig_w / args.image_size
        scale_y = orig_h / args.image_size
        boxes[:, 0::2] *= scale_x
        boxes[:, 1::2] *= scale_y

        # Draw box on test results
        draw = ImageDraw.Draw(img)
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=10)
            draw.text((x1, max(y1 - 20, 0)), f"paper {score:.2f}", fill="red")

        # Saving result images on result folder
        save_path = os.path.join(out_folder, f"result_{filename}")
        img.save(save_path)
        print(f"  {filename} → {len(boxes)} box | score: "
              f"{[f'{s:.2f}' for s in scores.tolist()]} → {save_path}")

if __name__ == "__main__":
    predict_images(
        image_folder = "./test_images",          
        model_path   = "./sessions/best_model.pth",
        score_threshold = 0.95                   # only take images with 95% paper-detect
    )