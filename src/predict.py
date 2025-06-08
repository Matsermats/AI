import argparse
import torch
from torchvision import transforms, models
from PIL import Image


CLASS_NAMES = None

def load_model(model_path, num_classes):
    # Load model without downloading pretrained weights to avoid network access
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)


def main(args):
    image = preprocess_image(args.image)
    model = load_model(args.model, len(args.class_names))
    with torch.no_grad():
        outputs = model(image)
        pred = outputs.argmax(dim=1).item()
    print(f"Prediction: {args.class_names[pred]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict class of an image")
    parser.add_argument("model", help="Path to trained model")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("class_names", nargs='+', help="List of class names in order")
    args = parser.parse_args()
    main(args)
