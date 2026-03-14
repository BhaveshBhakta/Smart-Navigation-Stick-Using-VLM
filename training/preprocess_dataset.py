import json
import os
from PIL import Image
from tqdm import tqdm

dataset_path = "dataset/train"
annotation_file = os.path.join(dataset_path, "annotations.json")

with open(annotation_file, "r") as f:
    data = json.load(f)

annotations = data["annotations"]
images = data["images"]

output_folder = "training/processed_dataset/images"
os.makedirs(output_folder, exist_ok=True)

pairs = []

LIMIT = 3000  # number of images to use

for i in tqdm(range(LIMIT)):

    image_name = images[i]["file_name"]
    caption = annotations[i]["caption"].lower()

    image_path = os.path.join(dataset_path, image_name)

    img = Image.open(image_path).convert("RGB")

    img = img.resize((384, 384))

    save_path = os.path.join(output_folder, image_name)

    img.save(save_path)

    pairs.append({
        "image": image_name,
        "caption": caption
    })

# save processed captions
with open("training/processed_dataset/captions.json", "w") as f:
    json.dump(pairs, f)

print("Preprocessing complete")
print("Total samples:", len(pairs))