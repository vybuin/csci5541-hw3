# HW3: Finetuning LLMs for text classification

## Installation
Make sure Python 3.11+ is installed, then run:
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

Here are two options to generate the required dataset files locally:

1. Run the following command in the terminal:

```bash
python load_data.py
```
2. Or open `load_data.py` in your IDE (e.g., VS Code) and press the Run button.

This script downloads the DeepfakeTextDetect dataset from Hugging Face (yaful/DeepfakeTextDetect), converts it into the required JSON format, and saves the files into the ./data/ directory.

## Implementation Notes
- **Batch size**: The batch size was set to 16 (default was 8) for faster training. This does not affect experimental validity since it is not a controlled variable.
- **Validation size**: The validation size was set to 2000 for faster training. This was done to reduce per-epoch evaluation time while maintaining sufficient sample size.
- **Padding**: `DataCollatorWithPadding` was used instead of `padding="max_length"` for dynamic padding to reduce training time.

## Running Experiments
Use the following naming convention for --best_model_name so plots work correctly:
### Epoch experiments
```bash
python main.py --train --file_folder ./data --epoch 1 --model_dir ./models/per_epoch --best_model_name epoch_1
python main.py --train --file_folder ./data --epoch 2 --model_dir ./models/per_epoch --best_model_name epoch_2
```

### Training size experiments
```bash
python main.py --train --file_folder ./data --epoch 4 --train_size 50 --model_dir ./models/per_size --best_model_name size_50
python main.py --train --file_folder ./data --epoch 4 --train_size 100 --model_dir ./models/per_size --best_model_name size_100
```

### Learning rate experiments
```bash
python main.py --train --file_folder ./data --epoch 4 --learning_rate 0.00001 --model_dir ./models/per_lr --best_model_name lr_0.00001
python main.py --train --file_folder ./data --epoch 4 --learning_rate 0.00002 --model_dir ./models/per_lr --best_model_name lr_0.00002
```

## General Testing & Plotting
```bash
python main.py model_name distilbert-base-uncased --file_folder ./data --model_dir ./models/per_epoch --plot_name epoch --test
python main.py model_name distilbert-base-uncased --file_folder ./data --model_dir ./models/per_size --plot_name size --test
python main.py model_name distilbert-base-uncased --file_folder ./data --model_dir ./models/per_lr --plot_name lr --test
python main.py file_folder ./data --model_dir ./best_model --best_model_name final_model --test
```
