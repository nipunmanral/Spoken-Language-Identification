import numpy as np
import glob
import os
from keras.models import Model
from keras.layers import Input, Dense, GRU, CuDNNGRU, CuDNNLSTM
from keras import optimizers
import h5py
from sklearn.model_selection import train_test_split
from keras.models import load_model


skipBlock1 = False

            
def language_name(index):
    if index == 0:
        return "English"
    elif index == 1:
        return "Hindi"
    elif index == 2:
        return "Mandarin"

# ---------------------------BLOCK 1------------------------------------
# COMMENT/UNCOMMENT BELOW CODE BLOCK -
# Below code extracts mfcc features from the files provided into a dataset
codePath = './train/'
num_mfcc_features = 64

# if set to true, will not do this process again
if (not skipBlock1):
    
    # reads the directories ("codepath/en/name.wav" for example) and extracs MFCCs beside it
    for dir in glob.glob(codePath + '*'):
        print("Creating numpy arrays for " + dir)
        for file in glob.glob(dir+'/*.wav'):
            fs, data = wavfile.read(file)

            arr = data[0:num_mfcc_features]
            np.save(file+".npy", arr)
            
          
    english_mfcc = np.array([]).reshape(0, num_mfcc_features)
    for file in glob.glob(codePath + 'english/*.npy'):
        current_data = np.load(file).T
        english_mfcc = np.vstack((english_mfcc, current_data))

    hindi_mfcc = np.array([]).reshape(0, num_mfcc_features)
    for file in glob.glob(codePath + 'hindi/*.npy'):
        current_data = np.load(file).T
        hindi_mfcc = np.vstack((hindi_mfcc, current_data))

    mandarin_mfcc = np.array([]).reshape(0, num_mfcc_features)
    for file in glob.glob(codePath + 'mandarin/*.npy'):
        current_data = np.load(file).T
        mandarin_mfcc = np.vstack((mandarin_mfcc, current_data))

    # Sequence length is 10 seconds
    sequence_length = 1000
    list_english_mfcc = []
    num_english_sequence = int(np.floor(len(english_mfcc)/sequence_length))
    for i in range(num_english_sequence):
        list_english_mfcc.append(english_mfcc[sequence_length*i:sequence_length*(i+1)])
    list_english_mfcc = np.array(list_english_mfcc)
    english_labels = np.full((num_english_sequence, 1000, 3), np.array([1, 0, 0]))

    list_hindi_mfcc = []
    num_hindi_sequence = int(np.floor(len(hindi_mfcc)/sequence_length))
    for i in range(num_hindi_sequence):
        list_hindi_mfcc.append(hindi_mfcc[sequence_length*i:sequence_length*(i+1)])
    list_hindi_mfcc = np.array(list_hindi_mfcc)
    hindi_labels = np.full((num_hindi_sequence, 1000, 3), np.array([0, 1, 0]))

    list_mandarin_mfcc = []
    num_mandarin_sequence = int(np.floor(len(mandarin_mfcc)/sequence_length))
    for i in range(num_mandarin_sequence):
        list_mandarin_mfcc.append(mandarin_mfcc[sequence_length*i:sequence_length*(i+1)])
    list_mandarin_mfcc = np.array(list_mandarin_mfcc)
    mandarin_labels = np.full((num_mandarin_sequence, 1000, 3), np.array([0, 0, 1]))

    del english_mfcc
    del hindi_mfcc
    del mandarin_mfcc

    total_sequence_length = num_english_sequence + num_hindi_sequence + num_mandarin_sequence
    Y_train = np.vstack((english_labels, hindi_labels))
    Y_train = np.vstack((Y_train, mandarin_labels))

    X_train = np.vstack((list_english_mfcc, list_hindi_mfcc))
    X_train = np.vstack((X_train, list_mandarin_mfcc))

    del list_english_mfcc
    del list_hindi_mfcc
    del list_mandarin_mfcc

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2)

    with h5py.File("mfcc_dataset.hdf5", 'w') as hf:
        hf.create_dataset('X_train', data=X_train)
        hf.create_dataset('Y_train', data=Y_train)
        hf.create_dataset('X_val', data=X_val)
        hf.create_dataset('Y_val', data=Y_val)
# ---------------------------------------------------------------


# --------------------------BLOCK 2-------------------------------------
# Load MFCC Dataset created by the code in the previous steps
with h5py.File("mfcc_dataset.hdf5", 'r') as hf:
    X_train = hf['X_train'][:]
    Y_train = hf['Y_train'][:]
    X_val = hf['X_val'][:]
    Y_val = hf['Y_val'][:]
# ---------------------------------------------------------------


# ---------------------------BLOCK 3------------------------------------
# Setting up the model for training
DROPOUT = 0.3
RECURRENT_DROP_OUT = 0.2
optimizer = optimizers.Adam(decay=1e-4)
main_input = Input(shape=(sequence_length, 64), name='main_input')

# ### main_input = Input(shape=(None, 64), name='main_input')
# ### pred_gru = GRU(4, return_sequences=True, name='pred_gru')(main_input)
# ### rnn_output = Dense(3, activation='softmax', name='rnn_output')(pred_gru)

layer1 = CuDNNLSTM(64, return_sequences=True, name='layer1')(main_input)
layer2 = CuDNNLSTM(32, return_sequences=True, name='layer2')(layer1)
layer3 = Dense(100, activation='tanh', name='layer3')(layer2)
rnn_output = Dense(3, activation='softmax', name='rnn_output')(layer3)

model = Model(inputs=main_input, outputs=rnn_output)
print('\nCompiling model...')
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
model.summary()
history = model.fit(X_train, Y_train, batch_size=32, epochs=75, validation_data=(X_val, Y_val), shuffle=True, verbose=1)
model.save('sld.hdf5')
# ---------------------------------------------------------------

# --------------------------BLOCK 4-------------------------------------
# Inference Mode Setup
streaming_input = Input(name='streaming_input', batch_shape=(1, 1, 64))
pred_layer1 = CuDNNLSTM(64, return_sequences=True, name='layer1', stateful=True)(streaming_input)
pred_layer2 = CuDNNLSTM(32, return_sequences=True, name='layer2')(pred_layer1)
pred_layer3 = Dense(100, activation='tanh', name='layer3')(pred_layer2)
pred_output = Dense(3, activation='softmax', name='rnn_output')(pred_layer3)
streaming_model = Model(inputs=streaming_input, outputs=pred_output)
streaming_model.load_weights('sld.hdf5')
# streaming_model.summary()
# ---------------------------------------------------------------

# ---------------------------BLOCK 5------------------------------------
# Language Prediction for a random sequence from the validation data set
random_val_sample = np.random.randint(0, X_val.shape[0])
random_sequence_num = np.random.randint(0, len(X_val[random_val_sample]))
test_single = X_val[random_val_sample][random_sequence_num].reshape(1, 1, 64)
val_label = Y_val[random_val_sample][random_sequence_num]
true_label = language_name(np.argmax(val_label))
print("***********************")
print("True label is ", true_label)
single_test_pred_prob = streaming_model.predict(test_single)
pred_label = language_name(np.argmax(single_test_pred_prob))
print("Predicted label is ", pred_label)
print("***********************")
# ---------------------------------------------------------------

# ---------------------------BLOCK 6------------------------------------
## COMMENT/UNCOMMENT BELOW
# Prediction for all sequences in the validation set - Takes very long to run
print("Predicting labels for all sequences - (Will take a lot of time)")
list_pred_labels = []
for i in range(X_val.shape[0]):
    for j in range(X_val.shape[1]):
        test = X_val[i][j].reshape(1, 1, 64)
        seq_predictions_prob = streaming_model.predict(test)
        predicted_language_index = np.argmax(seq_predictions_prob)
        list_pred_labels.append(predicted_language_index)
pred_english = list_pred_labels.count(0)
pred_hindi = list_pred_labels.count(1)
pred_mandarin = list_pred_labels.count(2)
print("Number of English labels = ", pred_english)
print("Number of Hindi labels = ", pred_hindi)
print("Number of Mandarin labels = ", pred_mandarin)
# ---------------------------------------------------------------
