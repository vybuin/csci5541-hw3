# HW3: Finetuning LLMs for text classification

## Installation
Make sure you have Python 3.11+ installed, then run:
```bash
pip install torch
pip install transformers -U
pip install accelerate -U
pip install datasets
pip install polars
pip install pandas
pip install numpy
pip install matplotlib
```
Or install all at once using the requirements.txt:
```bash
pip install -r requirements.txt
```

## Dataset

The dataset files (`TrainingData.json`, `ValidationData.json`, and `TestingData.json`) are not included in this repository due to their large file size.

To generate the required dataset files locally, you can either:

1. Run the following command in the terminal:

```bash
python load_data.py
```
2. Or open `load_data.py` in your IDE (e.g., VS Code) and press the Run button.

This script downloads the DeepfakeTextDetect dataset from Hugging Face (yaful/DeepfakeTextDetect), converts it into the required JSON format, and saves the files into the ./data/ directory.

## Implementation Notes
- **Batch size**: Set to 16(default was 8) for faster training. This does not affect experimental validity since it is not a controlled variable.
- **Padding**: Uses `DataCollatorWithPadding` instead of `padding="max_length"` for dynamic padding to reduce training time.

## Running Experiments
Use the following naming convention for --best_model_name so plots work correctly:
### Epoch experiments
```bash
python main.py --train --epoch 10 --model_dir ./models/per_epoch --best_model_name epoch_10
python main.py --train --epoch 11 --model_dir ./models/per_epoch --best_model_name epoch_11
```

### Training size experiments
```bash
python main.py --train --train_size 50 --model_dir ./models/per_size --best_model_name size_50
python main.py --train --train_size 100 --model_dir ./models/per_size --best_model_name size_100
```

### Learning rate experiments
```bash
python main.py --train --learning_rate 0.0001 --model_dir ./models/per_lr --best_model_name lr_0.0001
python main.py --train --learning_rate 0.001 --model_dir ./models/per_lr --best_model_name lr_0.001
```

## General Testing & Plotting
```bash
python main.py --test --model_dir ./models/per_epoch --plot_name epoch
python main.py --test --model_dir ./models/per_size --plot_name size
python main.py --test --model_dir ./models/per_lr --plot_name lr
```