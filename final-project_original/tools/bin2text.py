#!/home/s0/shpcta/miniconda3/envs/shpc24-fall/bin/python

# Description: This script converts binary files to text files for output visualization.
# Usage: python bin2text.py <input_path> <output_path>

import os
import sys
import torch
import datasets
import numpy as np
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import pickle

seed = 1234

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def tokenize_example(example, tokenizer, max_length):
    tokens = tokenizer(example["text"])[:max_length]
    return {"tokens": tokens}

# Decode tokenized text
def reverse_tokenize(ids, vocab):
    # Reverse the tokenization by converting ids back to tokens
    tokens = vocab.lookup_tokens(ids)
    sentence = " ".join(tokens).replace("<pad>", "").strip()  # Remove padding tokens
    return sentence

def load_inputs(filepath, num_sentences, seq_len):
    # Load inputs.bin and return the tokenized IDs as a list of lists
    with open(filepath, 'rb') as f:
        data = f.read()
        inputs = np.frombuffer(data, dtype=np.int32).reshape(num_sentences, seq_len)
    return inputs

def load_outputs(filepath):
    # Load outputs.bin and return the prediction values
    with open(filepath, 'rb') as f:
        data = f.read()
        outputs = np.frombuffer(data, dtype=np.float32).reshape(-1, 2)  # N x 2 predictions
    return outputs

def predict_sentiment_from_output(output):
    # Given a raw output (logits), return the predicted sentiment (0 for negative, 1 for positive) and its probability
    probability = torch.softmax(torch.tensor(output), dim=-1)
    predicted_class = probability.argmax().item()
    predicted_probability = probability[predicted_class].item()
    return predicted_class, predicted_probability

def main(inputs_filepath, outputs_filepath):
    
    """ Using pickle we don't need to run this code block

        # Get Tokenizer
        tokenizer = get_tokenizer("basic_english")

        # Load the train data from the IMDB dataset
        train_data = datasets.load_dataset("imdb", split="train")  # This loads the 'text' field

        max_length = 256
        train_data = train_data.map(
            tokenize_example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length}
        )

        valid_size = 0.25
        train_valid_data = train_data.train_test_split(test_size=valid_size)
        train_data = train_valid_data['train']
        valid_data = train_valid_data['test']

        min_freq = 5
        special_tokens = ["<unk>", "<pad>"]

        # Generate Vocab using tokenized 'text' field
        vocab = build_vocab_from_iterator(
            train_data["tokens"],
            min_freq=min_freq, 
            specials=special_tokens
        )

        unk_index = vocab["<unk>"]
        pad_index = vocab["<pad>"]
        vocab.set_default_index(unk_index)

        # Save Vocab from pickle file
        with open("./tools/vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    """
    
    # Load Vocab from pickle file 
    if os.path.exists("./tools/vocab.pkl"):
        vocab_path = "./tools/vocab.pkl"
    else:
        vocab_path = "./vocab.pkl"
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    num_sentences = 16384  # Assuming you know the number of sentences
    seq_len = 16

    # Load inputs and outputs
    inputs = load_inputs(inputs_filepath, num_sentences, seq_len)
    outputs = load_outputs(outputs_filepath)

    # Only use the first N inputs, where N is the number of outputs
    N = len(outputs)
    inputs = inputs[:N]

    # Check that the number of inputs matches the number of outputs
    assert len(inputs) == len(outputs), "Mismatch between number of inputs and outputs."

    # Process each sentence and prediction
    for i, (input_ids, output) in enumerate(zip(inputs, outputs), start=1):
        sentence = reverse_tokenize(input_ids, vocab)
        predicted_sentiment, predicted_probability = predict_sentiment_from_output(output)

        sentiment_label = "positive" if predicted_sentiment == 1 else "negative"
        
        print(f"Sentence #{i}")
        print(f" Input Sentence: {sentence}")
        print(f" Tokenized IDs: {input_ids.tolist()}") 
        print(f" Predicted Sentiment: {predicted_sentiment} ({sentiment_label})")
        print(f" Probability: {predicted_probability:.4f}")
        print()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tools/bin2text.py <input_path> <output_path>")
        print(" E.g., python tools/bin2text.py data/inputs.bin data/outputs.bin")
        print(" <input_path>: Path to the inputs.bin file")
        print(" <output_path>: Path to the outputs.bin file")
        sys.exit(1)

    inputs_filepath = sys.argv[1]
    outputs_filepath = sys.argv[2]

    main(inputs_filepath, outputs_filepath)
