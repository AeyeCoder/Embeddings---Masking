Sentiment Analysis on IMDB Reviews

This project demonstrates a simple sentiment analysis pipeline using TensorFlow and TensorFlow Datasets. The model classifies IMDB movie reviews as positive or negative.

Overview

Dataset: IMDB reviews (from tensorflow_datasets)

Preprocessing:

Text vectorization with a vocabulary size of 1,200

Automatic batching, shuffling, and prefetching

Model:

Embedding layer (128 dimensions) with masking support

GRU layer (128 units)

Dense output layer with sigmoid activation

Training:

Loss: Binary Cross-Entropy

Optimizer: Nadam

Metric: Accuracy

Epochs: 4 (can be increased with early stopping)

Code

The full training script is available in sentiment_analysis.py. Training is straightforward:

python sentiment_analysis.py

Results

After 4 epochs, the model already shows reasonable validation accuracy. Training for more epochs with early stopping will further improve results.

Next Steps

Tune vocab_size and embed_size

Add regularization (Dropout, L2)

Try Bi-directional GRU or LSTM

Compare with pre-trained sentence embeddings (e.g., Universal Sentence Encoder, BERT)
