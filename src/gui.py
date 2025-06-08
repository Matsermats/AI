import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms, models


def load_model(model_path, num_classes):
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
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


class ImageClassifierGUI(tk.Tk):
    def __init__(self, model_path, class_names):
        super().__init__()
        self.title("Image Classifier")
        self.configure(padx=20, pady=20)
        self.model = load_model(model_path, len(class_names))
        self.class_names = class_names

        self.image_label = tk.Label(self)
        self.image_label.pack(pady=10)

        self.prediction_var = tk.StringVar()
        self.prediction_var.set("Select an image to classify")
        tk.Label(self, textvariable=self.prediction_var, font=("Helvetica", 14)).pack(pady=10)

        tk.Button(self, text="Open Image", command=self.open_image).pack()

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        try:
            img = Image.open(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {e}")
            return
        img_resized = img.resize((224, 224))
        photo = ImageTk.PhotoImage(img_resized)
        self.image_label.configure(image=photo)
        self.image_label.image = photo
        self.classify_image(path)

    def classify_image(self, path):
        image = preprocess_image(path)
        with torch.no_grad():
            outputs = self.model(image)
            pred = outputs.argmax(dim=1).item()
        self.prediction_var.set(f"Prediction: {self.class_names[pred]}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Launch simple GUI for image classification"
    )
    parser.add_argument(
        "--model",
        default="model.pth",
        help="Path to trained model (default: model.pth)",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["dog", "cat", "other"],
        help="List of class names in order",
    )
    args = parser.parse_args()

    gui = ImageClassifierGUI(args.model, args.class_names)
    gui.mainloop()


if __name__ == "__main__":
    main()