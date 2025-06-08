import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import sys
from pathlib import Path
import torch
from torchvision import transforms, models


def load_model(model_path, num_classes):
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


class ImageClassifierGUI(tk.Tk):
    def __init__(self, model_path, class_names):
        super().__init__()
        self.title("🐶🐱 AI Beeldclassificatie")
        self.geometry("520x620")
        self.configure(bg="#eeeeee")

        self.model = load_model(model_path, len(class_names))
        self.class_names = class_names

        # === Frame voor afbeelding ===
        self.image_frame = tk.Frame(self, bg="#ffffff", width=320, height=300, relief="groove", borderwidth=2)
        self.image_frame.pack(pady=20)
        self.image_label = tk.Label(self.image_frame, bg="#ffffff")
        self.image_label.pack(padx=10, pady=10)

        self.file_var = tk.StringVar()
        self.file_label = tk.Label(
            self.image_frame,
            textvariable=self.file_var,
            bg="#ffffff",
            font=("Helvetica", 12, "italic"),
        )
        self.file_label.pack(pady=(0, 5))

        # === Frame voor voorspelling ===
        self.prediction_frame = tk.Frame(self, bg="#f0f0f0", relief="ridge", borderwidth=2)
        self.prediction_frame.pack(pady=10, fill="x", padx=30)

        self.prediction_var = tk.StringVar()
        self.prediction_var.set("🖼️ Selecteer een afbeelding om te classificeren")
        self.prediction_label = tk.Label(
            self.prediction_frame,
            textvariable=self.prediction_var,
            font=("Helvetica", 18, "bold"),
            fg="#007acc",
            bg="#f0f0f0",
            wraplength=400,
        )
        self.prediction_label.pack(pady=15)

        # === Knop om afbeelding te openen ===
        self.open_button = tk.Button(
            self, text="📂 Open afbeelding", font=("Helvetica", 13),
            command=self.open_image, bg="#4CAF50", fg="white", padx=10, pady=5
        )
        self.open_button.pack(pady=15)

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Afbeeldingen", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        try:
            img = Image.open(path)
            img_resized = img.resize((300, 300))
            photo = ImageTk.PhotoImage(img_resized)
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # referentie bewaren
            self.file_var.set(f"Bestand: {os.path.basename(path)}")
            print(f"Gekozen afbeelding: {path}")
            self.classify_image(path)
        except Exception as e:
            messagebox.showerror("Fout", f"Afbeelding kan niet geopend worden:\n{e}")

    def classify_image(self, path):
        try:
            image = preprocess_image(path)
            with torch.no_grad():
                outputs = self.model(image)
                pred = outputs.argmax(dim=1).item()
            label = self.class_names[pred].capitalize()
            is_dog = label.lower() == "dog"
            color = "#4CAF50" if is_dog else "#ff4081"
            emoji = "🐶" if is_dog else "🐱"
            self.prediction_label.configure(fg=color)
            self.prediction_var.set(f"{emoji} Voorspelling: {label}")
            print(f"Voorspelling voor {os.path.basename(path)}: {label}")
        except Exception as e:
            messagebox.showerror("Fout", f"Classificatie mislukt:\n{e}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / "model.pth"
    class_names = ["cat", "dog"]  # Zorg dat dit overeenkomt met je trainingsdata

    if not model_path.exists():
        print(f"❌ Modelbestand '{model_path}' bestaat niet.")
        sys.exit(1)

    gui = ImageClassifierGUI(str(model_path), class_names)
    gui.mainloop()
