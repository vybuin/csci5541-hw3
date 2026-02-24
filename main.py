"""
DistilBERT model from Hugging Face
https://huggingface.co/docs/transformers/en/model_doc/distilbert
"""

import os
import argparse
from typing import List, Union

import numpy as np
import pandas as pd
import polars as pl

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

def load_model_and_tokenizer(model_name:str="distilbert-base-uncased", 
                             num_labels:int=2):
    """
    Function to load model and tokenizer based on model_name and num_labels for classification

    Args:
        model_name: Name of the model (as loadable from Transformers library)
        num_labels: Number of classification labels (e.g., positive, negative -> 2)

    Returns:
        tokenizer: loaded tokenizer
        model: loaded model
    """
    print(f"Loading model and tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    print("Model and tokenizer loaded.")
    return tokenizer, model
    

def tokenize_function(tokenizer,
                      examples:dict,):
    """
    Function to tokenize the 'sentence' column
    Padding is set to max_length to pad all sentences to the same length
    Trunction is set to true to cut sentences that are too long

    Args:
        tokenizer: AutoTokenizer to use to do the tokenization task
        examples: Dictionary format containing of sentence column

    Return:
        Tokenized output
    """
    # DONE: return tokenized output
    texts = examples.get("sentence")
    return tokenizer(texts, padding="max_length", truncation=True)

def convert_df_to_dataset(df_train: pd.DataFrame, 
                          df_valid: pd.DataFrame) -> DatasetDict:
    """
    Function to training, validation, and test Pandas DataFrames into a Hugging Face DatasetDict.
    
    Args:
        df_train: Pandas DataFrame for training.
        df_valid: Pandas DataFrame for validation.
        
    Return:
        dataset_dict: Dataset with train, validation, test split
    """
    # DONE: Convert each DataFrame to a Hugging Face Dataset
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_ds = Dataset.from_pandas(df_train)
    valid_ds = Dataset.from_pandas(df_valid)

    # DONE: Combine them into a single DatasetDict
    dataset_dict = DatasetDict({
        "train": train_ds,
        "validation": valid_ds
    })

    return dataset_dict

def compute_metrics(eval_pred):
    """
    Computes metric in the form that will be called by Trainer during evaluation

    Args:
        eval_pred: prediction output from the model

    Returns:
        dictionary in form "accuracy": accuracy
    """
    logits, labels = eval_pred

    # Get the predictions by finding the class with the highest logit
    predictions = np.argmax(logits, axis=-1)
    accuracy = get_accuracy(list_gt=labels,
                            list_pred=predictions)
    
    return {"accuracy": accuracy}

def train_model(model,
                train_dataset, valid_dataset,
                num_epochs: int,
                learning_rate: float,
                output_dir: str = './models',
                logging_dir: str = './logs',
                bestmodel_dir: str = 'mybestmodel') -> None:
    """
    Train model using HuggingFace Trainer model

    Args:
        num_epochs: number of training epochs
        output_dir: directory to save all models
        logging_dir: directory save the log files
        bestmodel_dir: directory to save the best performing model
    """
    training_args = TrainingArguments(
        output_dir=output_dir,           # Directory to save the model
        num_train_epochs=num_epochs,     # Total number of training epochs
        learning_rate=learning_rate,     # Learning rate
        per_device_train_batch_size=8,   # Batch size for training
        per_device_eval_batch_size=8,    # Batch size for evaluation
        logging_dir=logging_dir,         # Directory for to store logs
        logging_steps=50,                # Log every 50 steps
        eval_strategy="epoch",           # Run evaluation at the end of each epoch
        save_strategy="epoch",           # Save the model at the end of each epoch
        load_best_model_at_end=True,     # Load the best model found during training
    )

    # Initializing Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    # Start training
    print("Starting training")
    trainer.train()

    # Save best performing model
    print(f"\nBest performing model being saved to {os.path.join(output_dir, bestmodel_dir)}")
    trainer.save_model(os.path.join(output_dir, bestmodel_dir))

    # After training, get the final evaluation results
    print("\nTraining finished. Evaluating model on validation set")

    eval_results = trainer.evaluate()
    print(f"\nFinal Evaluation Results:\n{eval_results}")

def load_model(model_dir:str,
               model_name:str="distilbert-base-uncased"):
    """
    Function to load trained model

    Args:
        model_dir: directory with saved model
        model_name: name of the base model (as used when training input)

    Returns:
        tokenizer: loaded tokenizer
        model: loaded model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    return tokenizer, model

def predict_sentiment(tokenizer, model,
                      text:str) -> int:
    """
    Function to predict sentiment of the given text

    Args:
        tokenizer: loaded tokenizer
        model: loaded model
        text: input text to do sentiment analysis

    Return:
        predicted class
    """
    # DONE: Tokenize the input text
    inputs = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # DONE: Get model output
    model.eval() #set model to evaluation mode
    with torch.no_grad():
        outputs = model(**inputs) #forward operation to get the logits

    # DONE: Get the predicted class ID
    predict_class_id = torch.argmax(outputs.logits, dim=-1).item() #get the ind of the highest logit

    # DONE: Return prediction
    return int(predict_class_id)

def decide_train_size(pd_train,
                      train_size:int) -> pd.DataFrame:
    """
    Input: The full training dataset and a target train_size (e.g., 400)
    Task: This function must return a subset of the original dataset containing exactly train_size samples.
    """
    # DONE: To fill
    if train_size >= len(pd_train):
        return pd_train.reset_index(drop=True)
    
    subset = pd_train.sample(n=train_size, random_state = 42)
    return subset.reset_index(drop=True)

def get_accuracy(list_gt: np.array, 
                 list_pred: np.array):
    """
    Input: A list of ground truth labels, and a list of predicted labels
    Task: Calculate the accuracy of the model based on the ground truth label list
    """
    # DONE: To fill
    gt = np.asarray(list_gt) 
    pred = np.asarray(list_pred)
    if gt.size == 0:
        return 0.0
    return float((gt == pred).mean())

def plot_result(plot_name:str, data:dict) -> None:
    """
    Input: Decide on the input that will be relevant to completing the task
    Task: Using a library like matplotlib, you must create two plots:
        Plot 1 (Epochs): Create a line graph with
            X-axis: number of Epochs (e.g., 1, 2, 3, ..., 10); minimum of 10 different epoch sizes required
            Y-axis: validation accuracy (i.e., accuracy on validation set) and test accuracy (i.e., accuracy on test set)
            Save as: per_epoch.png
        Plot 2 (Training Data Size): Create a line graph with
            X-axis: training data size (e.g., 50, 100, 400, 800); minimum of 4 different training data sizes required
            Y-axis: validation accuracy (i.e., accuracy on validation set) and test accuracy (i.e., accuracy on test set)
            Save as: per_size.png
        Plot 3 (Learnig Rate): Create a line graph with
            X-axis: training data size (e.g., 0.0001, 0.001, 0.01); minimum of 3 different  learning rates required
            Y-axis: validation accuracy (i.e., accuracy on validation set) and test accuracy (i.e., accuracy on test set)
            Save as: per_learning_rate.png   
    """
    import os
    import matplotlib.pyplot as plt

    # DONE: To fill
    x_axis = data["x"] 
    y_axis_valid = data["y_val_acc"]
    #y_axis_test = data["y_test_acc"] DOESNT NEED TEST ACCURACY FOR THE PLOT

    # input data dict will look like this
    # data = {
    #     "x": list_model_names, 
    #     "y_val_acc": list_valid_score,
    # }

    out_dir = "./plots"
    os.makedirs(out_dir, exist_ok=True)

    if plot_name == "epoch":
        xlabel = "Epochs"
        out_file = "per_epoch.png"

    elif plot_name == "size":
        xlabel = "Training Data Size"
        out_file = "per_size.png"

    elif plot_name == "lr" or plot_name == "learning_rate":
        xlabel = "Learning Rate"
        out_file = "per_learning_rate.png"

    else:
        xlabel = "x"
        out_file = "plot.png"

    plt.figure()
    plt.plot(x_axis, y_axis_valid, label="Validation Accuracy", marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("Validation Accuracy")
    plt.title(f"Validation Accuracy vs {xlabel}")
    plt.xticks(x_axis)
    plt.xscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, out_file))
    plt.close()
    print(f"Saved {out_file}")


def test(model_name:str, model_dir:str,
         pl_data: pl.DataFrame):

    tokenizer, model = load_model(model_dir=model_dir, model_name=model_name)

    pl_data = pl_data.with_columns(
        prediction = pl.col('sentence').map_elements(lambda x: predict_sentiment(tokenizer, model, x))
    )

    return get_accuracy(pl_data['label'].to_list(), pl_data['prediction'].to_list())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sentiment analysis program with distilbert')

    parser.add_argument('--model_name', default='distilbert-base-uncased', 
                        help='Model to train for sentiment analysis')
    parser.add_argument('--file_folder', default='./',
                        help='Folder with train, validation, test dataset')
    parser.add_argument('--epoch', type=int, default=2, 
                        help='Number of epochs to train the model for')
    parser.add_argument('--train_size', type=int, default=800,
                        help='Size of train datasize to use to train the model')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                    help='Learning rate for optimizer')
    
    parser.add_argument('--model_dir', default='./models',
                        help='Name of the folder to save all models')
    parser.add_argument('--best_model_name',
                        help='Name of the folder to save the best model')
    parser.add_argument('--plot_name', default='./test.png', #change to ./plots.png??, check hw3 instrctions
                        help='Name of the plot')
    
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--test', action='store_true',
                        help='Test the model')

    args = parser.parse_args()

    if args.train:
        # Load training, valid, and test dataset as polars dataframe
        pl_train = pl.read_ndjson(os.path.join(args.file_folder, 'TrainingData.json')).to_pandas()
        pl_train = decide_train_size(pl_train, args.train_size)         # setting training size
        pl_valid = pl.read_ndjson(os.path.join(args.file_folder, 'ValidationData.json')).to_pandas()
        
        #Add the line below for testing purposes to make training fatser
        #pl_valid = pl_valid.sample(n=200, random_state=42).reset_index(drop=True) 

        # Create dataset dictionary
        dataset = convert_df_to_dataset(pl_train, pl_valid)

        # Load model and tokenizer
        tokenizer, model = load_model_and_tokenizer(model_name=args.model_name)

        # Tokenize the created dataset
        tokenized_datasets = dataset.map(
            lambda x: tokenize_function(tokenizer, x), 
            batched=True)
        
        train_dataset = tokenized_datasets['train']
        valid_dataset = tokenized_datasets['validation']
        print(f"Using {len(train_dataset)} samples for training and {len(valid_dataset)} for validation.")

        # Training the model
        train_model(model=model,
                    train_dataset=train_dataset, valid_dataset=valid_dataset,
                    num_epochs=args.epoch, learning_rate=args.learning_rate,
                    output_dir=args.model_dir, bestmodel_dir=(args.best_model_name or 'mybestmodel'))
        
        # Mini sample output (just to check)
        pos_sentence = "This movie was fantastic, I really loved it."   # Expected: 1
        neg_sentence = "It was a complete waste of time and money."     # Expected: 0

        print(f"Sentence: '{pos_sentence}' -> Sentiment: {predict_sentiment(tokenizer, model, pos_sentence)}")
        print(f"Sentence: '{neg_sentence}' -> Sentiment: {predict_sentiment(tokenizer, model, neg_sentence)}")

    elif args.test:
        pl_valid = pl.read_ndjson(os.path.join(args.file_folder, 'ValidationData.json'))
        pl_test = pl.read_ndjson(os.path.join(args.file_folder, 'TestingData.json'))

        if args.best_model_name:
            # If best_model_name argument is given, then append that to the path and run the test mode
            model_dir = os.path.join(args.model_dir, args.best_model_name)
            test_accuracy = test(model_name=args.model_name, model_dir=model_dir,
                                 pl_data=pl_test)
        else:
            list_model_names = []
            list_valid_score = []
            list_test_score = []
            # If best_model_name argument is not given, then iteratively load all models in the directory
            for md in os.listdir(args.model_dir):
                model_dir = os.path.join(args.model_dir, md)

                valid_accuracy = test(model_name=args.model_name, model_dir=model_dir,
                                      pl_data=pl_valid)
                test_accuracy = test(model_name=args.model_name, model_dir=model_dir,
                                     pl_data=pl_test)
                
                list_model_names.append(md)
                list_valid_score.append(valid_accuracy)
                list_test_score.append(test_accuracy)

            # Visualize the score per model
            # TODO: Add lines to map the model folder names to be something representing the x axis of the plot
            # DONE: or do something to your local folder in the first place?

            #convert folder names like epoch_10/size_400/lr_0.001 -> numeric x values for the plot
            x_vals = []
            for name in list_model_names:
                #take the last chunk after underscore
                x_vals.append(float(name.split("_")[-1]))

            #sort by x so the line chart is ordered
            order = np.argsort(np.array(x_vals))
            x_sorted = list(np.array(x_vals)[order])
            val_sorted = list(np.array(list_valid_score)[order])

            data = {
                "x": x_sorted,
                "y_val_acc": val_sorted,
            }

            plot_result(args.plot_name, data)

    else:
        print("Need to select either train or test option.")