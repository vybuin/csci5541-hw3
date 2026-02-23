# HW3: Finetuning LLMs for text classification
Fine-tune a pretrained model to perform text classification. You will evaluate how three key factors (training dataset size, number of training epochs, and learning rate) impact the final performance of your model. Then construct the "Best Model" which incorporates the optimal parameters. 

## Dataset

The dataset files (`TrainingData.json`, `ValidationData.json`, and `TestingData.json`) are not included in this repository due to their large file size.

To generate the required dataset files locally, you can either:

1. Run the following command in the terminal:

```bash
python load_data.py
```
2. Or open load_data.py in your IDE (e.g., VS Code) and press the Run button.

This script downloads the DeepfakeTextDetect dataset from Hugging Face (yaful/DeepfakeTextDetect), converts it into the required JSON format, and saves the files into the ./data/ directory.
