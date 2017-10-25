import json, h5py
import numpy as np
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers.wrappers import Bidirectional
from keras.layers import Activation, Dense, Dropout, Embedding, GRU, LSTM, RepeatVector, TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint
from dataset import dataGenerator

input_filepath = 'savedata/'

# Parameters for the neural model and dataset

MAX_INPUT = 22          # 12 for basic seq2seq, same as max_len for attn
MAX_LENGTH = 22         # max sequence length in characters (for padding/truncating)
HIDDEN_SIZE = 128       # LSTM Nodes/Features/Dimension
EMBEDDING_SIZE = 128
BATCH_SIZE = 64
DROPOUTRATE = 0.3
LAYERS = 2              # bi-LSTM-RNN layers (not working)
MAX_EPOCHS = 10         # max iterations, early stop condition below

# parameters for model saving etc
model_filepath = "model/TEMP_seq2seq_weights-{epoch:02d}-{val_acc:.2f}.hdf5"  # for temp models
model_logdir = "model/"  # for t-board log files (not working atm)
stop_monitor = 'val_acc'  # variable for early stop: val_loss or val_acc
stop_delta = 0.01         # minimum delta before early stop (default = 0)
stop_epochs = 1           # how many epochs to do after stop condition (default = 0)


print('Loading data...\n')

x_words = [str(w) for w in list(np.load('encoded/x_words.npy'))]
y_words = [str(w) for w in list(np.load('encoded/y_words.npy'))]
x_vocab = [str(w) for w in list(np.load('encoded/x_vocab.npy'))]
y_vocab = [str(w) for w in list(np.load('encoded/y_vocab.npy'))]

kor2idx = np.load('encoded/kor2idx.npy').item()
idx2kor = np.load('encoded/idx2kor.npy').item()
eng2idx = np.load('encoded/eng2idx.npy').item()
idx2eng = np.load('encoded/idx2eng.npy').item()

train_idx = np.load('encoded/train_idx.npy')
test_idx = np.load('encoded/test_idx.npy')

x_train = np.load('encoded/x_train.npy')
x_test = np.load('encoded/x_test.npy')
y_train = np.load('encoded/y_train.npy')
y_test = np.load('encoded/y_test.npy')

VOCAB_KOR = len(kor2idx.keys())
VOCAB_ENG = len(eng2idx.keys())

print("zero-padding sequences...\n")
x_train = sequence.pad_sequences(x_train, maxlen=MAX_INPUT, truncating='post', padding='post')
x_test = sequence.pad_sequences(x_test, maxlen=MAX_INPUT, truncating='post', padding='post')
y_train = sequence.pad_sequences(y_train, maxlen=MAX_LENGTH, truncating='post', padding='post')
y_test = sequence.pad_sequences(y_test, maxlen=MAX_LENGTH, truncating='post', padding='post')

print("building model...\n")

model = Sequential()

encRNN = GRU
decRNN = LSTM

model.add(Embedding(VOCAB_KOR, EMBEDDING_SIZE, input_length=MAX_INPUT, mask_zero=True))
model.add(Dropout(DROPOUTRATE))
model.add(Bidirectional(encRNN(HIDDEN_SIZE, return_sequences=True)))
# model.add(Bidirectional(decRNN(HIDDEN_SIZE, return_sequences=True)))

# from attention import Attention
# model.add(Bidirectional(encRNN(HIDDEN_SIZE, return_sequences=True)))
# model.add(Attention())

model.add(Bidirectional(encRNN(HIDDEN_SIZE)))
model.add(RepeatVector(MAX_LENGTH))

model.add(Bidirectional(decRNN(HIDDEN_SIZE, return_sequences=True)))
model.add(Bidirectional(decRNN(HIDDEN_SIZE, return_sequences=True)))
model.add(Bidirectional(decRNN(HIDDEN_SIZE, return_sequences=True)))
model.add(TimeDistributed(Dense(VOCAB_ENG)))
model.add(Activation('softmax'))

# todo: load weights
# model.load_weights(model_weights_filename)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint(model_filepath,
                             monitor=['acc'])
earlystop = EarlyStopping(monitor=stop_monitor,
                          min_delta=stop_delta,
                          patience=stop_epochs,
                          verbose=1,
                          mode='auto')

callbacks_list = [checkpoint]

model.summary()

# history = model.fit(X_train, y_train,
#                       validation_data=(X_test, y_test),
#                       batch_size=BATCH_SIZE,
#                       epochs=MAX_EPOCHS,
#                       callbacks=callbacks_list
#                       )


generator = dataGenerator(BATCH_SIZE,
                        xfile=x_train,
                        yfile=y_train,
                        vocabsize=VOCAB_ENG,
                        epochsize=int(len(x_train)/BATCH_SIZE)-1)

# evaluate
egenerator = dataGenerator(BATCH_SIZE,
                           xfile=x_test,
                           yfile=y_test,
                           vocabsize=VOCAB_ENG,
                           epochsize=int(len(x_test)/BATCH_SIZE)-1)

history = model.fit_generator(generator, steps_per_epoch=int(len(x_train)/BATCH_SIZE)-1,
                              validation_data=egenerator,
                              validation_steps=int(len(x_test)/BATCH_SIZE)-1,
                              callbacks=callbacks_list,
                              epochs=MAX_EPOCHS)

hist_dict = history.history

# todo: just save to json one time
with open('model/json_attn_model.json', 'w') as f:
    json.dump(model.to_json(), f)
print("saved JSON model file to disk\n")
model.save('model/full_attn_model.h5')
print("saved model to disk\n")
model.save_weights('model/attn_model_weights.h5')
print("saved model weights to disk\n")
np.save('model/attn_hist_dict.npy', hist_dict)
print("saved history dictionary to disk\n")

scores = model.evaluate_generator(egenerator,
                                  steps=int(len(x_test)/BATCH_SIZE)-1
                                  )
print('')
print('Eval model...')
print("Accuracy: %.2f%%" % (scores[1] * 100), '\n')
print('')
