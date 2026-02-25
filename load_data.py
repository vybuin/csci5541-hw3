"""
Option 4
Task: Human vs. AI text detection
Dataset: https://huggingface.co/datasets/yaful/DeepfakeTextDetectLinks to an external site.
"""

from datasets import load_dataset

ds = load_dataset("yaful/DeepfakeTextDetect")

def normalize(example):
    return {
        "sentence": example["text"],
        "label": int(example["label"])
    }

train = ds["train"].map(normalize, remove_columns=ds["train"].column_names)
valid = ds["validation"].map(normalize, remove_columns=ds["validation"].column_names)
test  = ds["test"].map(normalize, remove_columns=ds["test"].column_names)

train.to_json("./data/TrainingData.json", orient="records", lines=True)
valid.to_json("./data/ValidationData.json", orient="records", lines=True)
test.to_json("./data/TestingData.json", orient="records", lines=True)