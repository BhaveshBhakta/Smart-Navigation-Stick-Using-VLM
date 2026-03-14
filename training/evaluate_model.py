import json
import os
from PIL import Image

from transformers import BlipProcessor, BlipForConditionalGeneration

import evaluate


dataset_folder = "training/processed_dataset"
image_folder = os.path.join(dataset_folder, "images")

caption_file = os.path.join(dataset_folder, "captions.json")


with open(caption_file) as f:
    data = json.load(f)


processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "blip_navigation_model"
)


bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")


predictions = []
references = []


LIMIT = 200


for i in range(LIMIT):

    item = data[i]

    image_path = os.path.join(image_folder, item["image"])

    image = Image.open(image_path).convert("RGB")

    inputs = processor(image, return_tensors="pt")

    out = model.generate(
        **inputs,
        max_new_tokens=30,
        num_beams=5,
        early_stopping=True
    )

    caption = processor.decode(out[0], skip_special_tokens=True)

    predictions.append(caption)

    references.append([item["caption"]])


bleu_score = bleu.compute(predictions=predictions, references=references)

meteor_score = meteor.compute(predictions=predictions, references=[r[0] for r in references])

rouge_score = rouge.compute(predictions=predictions, references=[r[0] for r in references])


print("\nEvaluation Results\n")

print("BLEU:", bleu_score["bleu"])
print("METEOR:", meteor_score["meteor"])
print("ROUGE-L:", rouge_score["rougeL"])