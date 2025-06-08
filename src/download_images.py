from duckduckgo_search import DDGS
import requests
import os

# 📂 Bepaal pad relatief aan script
base_dir = os.path.dirname(__file__)
cat_folder = os.path.join(base_dir, "DATA", "cat")
dog_folder = os.path.join(base_dir, "DATA", "dog")

def download_images(query, folder, max_images=30):
    os.makedirs(folder, exist_ok=True)
    with DDGS() as ddgs:
        results = ddgs.images(query, max_results=max_images)
        for i, result in enumerate(results):
            url = result["image"]
            try:
                img_data = requests.get(url, timeout=10).content
                file_path = os.path.join(folder, f"{query}_{i}.jpg")
                with open(file_path, "wb") as f:
                    f.write(img_data)
                print(f"✅ Saved: {file_path}")
            except Exception as e:
                print(f"⚠️ Error {i}: {e}")

# 📥 Download naar juiste mappen binnen project
download_images("cat", cat_folder, max_images=30)
download_images("dog", dog_folder, max_images=30)
