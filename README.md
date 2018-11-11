# Music Genre Classifier with Fourier Tranformation and Spectrograms

## About
- This is a music genre classifier written in Python.
- As the title suggests, the goal is to classify music files into 10 predetermined genres such as jazz, classical, country, pop, rock, and metal using GTZAN dataset, which is frequently used to benchmark music genre classification tasks.
- Fourier transformation along with spectrograms are used to do the same.

## How to run
This code was written and tested on python3.6. It might run on older versions as well. Please check 'requirements.txt' for all dependencies.

To train a model based on 
- spectrograms run `python main.py`.
- fourier tranformation only `python fourier.py`.

## Problem Description
Your task is to classify music files into 10 predetermined genres such as jazz, classical, country, pop, rock, and metal.

We will use the GTZAN dataset, which is frequently used to benchmark music genre classification tasks. It is organized into 10 distinct genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock. The dataset contains the first 30 seconds of 100 songs per genre. You can download the dataset from UNM Learn. In the folder genres you will find 10 folders, one per genre and 90 songs per folder. Additionally, you’ll find a validation folder with 100 unlabeled songs. The tracks are recorded at 22,050 Hz (22,050 readings per second) mono in the au format.

Design a learning experiment capable of predicting music genres given their audio.

[Here](https://www.kaggle.com/c/project-1-music-classifier-cs529-2018) is the link of the competition on Kaggle.
