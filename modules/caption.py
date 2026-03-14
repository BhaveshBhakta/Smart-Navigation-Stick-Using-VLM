from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load device 
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

def generate_caption(image_path):
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Preprocess image
    inputs = processor(image, return_tensors="pt").to(device)

    # Generate caption
    out = model.generate(
        **inputs,
        max_new_tokens=30,
        num_beams=5,
        early_stopping=True
    )

    # Decode caption
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption