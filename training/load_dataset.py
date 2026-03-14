import json
import os
from PIL import Image

dataset_path = "dataset/train"

annotation_file = os.path.join(dataset_path, "annotations.json")

with open(annotation_file, "r") as f:
    data = json.load(f)

annotations = data["annotations"]
images = data["images"]

print("Total annotations:", len(annotations))
print("Total images:", len(images))

os.makedirs("training/sample_images", exist_ok=True)

for i in range(10):

    image_info = images[i]
    image_name = image_info["file_name"]

    caption = annotations[i]["caption"]

    image_path = os.path.join(dataset_path, image_name)

    img = Image.open(image_path)

    save_path = f"training/sample_images/sample_{i}.jpg"

    img.save(save_path)

    print("Saved:", save_path)
    print("Caption:", caption)