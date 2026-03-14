import json
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import TrainingArguments, Trainer


# paths
dataset_folder = "training/processed_dataset"
image_folder = os.path.join(dataset_folder, "images")
caption_file = os.path.join(dataset_folder, "captions.json")


with open(caption_file) as f:
    pairs = json.load(f)


processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "blip_navigation_model/checkpoint-2250"
)


class CaptionDataset(Dataset):

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        item = self.pairs[idx]

        image_path = os.path.join(image_folder, item["image"])
        caption = item["caption"]

        image = Image.open(image_path).convert("RGB")

        inputs = processor(
            image,
            caption,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = inputs["input_ids"].clone()

        inputs["labels"] = labels

        inputs = {k: v.squeeze() for k, v in inputs.items()}

        return inputs


dataset = CaptionDataset(pairs)


training_args = TrainingArguments(

    output_dir="blip_navigation_model",

    per_device_train_batch_size=4,

    num_train_epochs=3,

    learning_rate=5e-5,

    fp16=True,

    logging_steps=50,

    save_steps=500,

    remove_unused_columns=False

)


trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=dataset

)


trainer.train()

trainer.save_model("blip_navigation_model")

print("Training complete")