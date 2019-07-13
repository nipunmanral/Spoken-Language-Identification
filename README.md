# Spoken Language Identification

## Objective
Spoken Language Identification (LID) is broadly defined as recognizing the language of a given speech utterance. It has numerous applications in automated language and speech recognition, multilingual machine translations, speech-to-speech translations, and emergency call routing. In this project, we will try to classify three languages (English, Hindi and Mandarin) from the spoken utterances that have been crowd-sourced. We will implement a GRU/LSTM model, and train it to classify the languages using Keras. We will use MFCC features as they are widely employed in various speech processing applications including LID.

## Environment Setup
Download the codebase and open up a terminal in the root directory. Make sure python 3.6 is installed in the current environment. Then execute

    pip install -r requirements.txt

This should install all the necessary packages for the code to run.

## Dataset
The dataset has a bunch of wav files and a csv file containing labels. The wav file names are anonymized, and class labels are provided as integers. Training is done with the provided integer class labels. The following mapping is used to convert language IDs to integer labels:
mapping = dict{’english ’: 0, ’hindi ’: 1, ’mandarin’: 2}

# Sample length
The full audio files are ∼ 10 minutes long which might be too long to train an RNN. Multiple 10 seconds samples are created from every utterance and the same label as the original utterance are assigned to them. The choice of sequence length can be changed to experiment with samples of different length.

## Audio Format
The wav files have 16KHz sampling rate, single channel, and 16-bit Signed Integer PCM encoding.

## Notes about the code
Thee code has been divided into 6 blocks. Kindly refer to the following notes to comment/uncomment the blocks as needed

- The code in Block 1 is used to extract the mfcc features provided and write them into a dataset “mfcc_dataset.hdf5”. This part of the code can be commented out if the hdf5 file already exists.

- The code in Block 2 is used to read the “mfcc_dataset.hdf5” dataset. Do not comment it out.

- The code in Block 3 is used to train the model. Comment it out after the model has been trained and saved by the name “sld.hdf5”

- The code in Block 4 sets up the inference mode.

- The code in Block 5 runs the streaming model in inference mode by predicting the label for a single random sequence from the validation dataset.

- The code in Block 6 runs the streaming model in inference mode by predicting the the labels for all the sequences in the validation dataset. Comment this out since it can take a long time to run.
