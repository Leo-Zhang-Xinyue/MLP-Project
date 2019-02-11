import keras.backend as K
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import re
import itertools
import numpy as np


from time import time
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda
from keras.optimizers import Adadelta
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences


def text_to_word_list(text):
    ''' Pre process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text



#read the raw data by pandas
file_read = pd.read_csv('quora_duplicate_questions.tsv', sep = '\t', header = 0,index_col='id')

#stopwords in nltk corpus
nltk.download('stopwords')
stops = set(nltk.corpus.stopwords.words('english'))

#pretrained model of the word emeddings
word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

#convert the question text into lists of word indices
vocabulary = {}
idx_voc = []
questions = ['question1', 'question2']
for index, row in file_read.iterrows():
    for question in questions:
        q2i = []
        for word in text_to_word_list(row[question]):
            if word in stops and word not in word2vec.vocab:
                continue
            if word not in vocabulary:
                vocabulary[word] = len(idx_voc)
                idx_voc.append(word)
                q2i.append(len(idx_voc))
            else:
                q2i.append(vocabulary[word])
        file_read = file_read.set_value(index, question, q2i)

embedding_dim = 300
embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)
embeddings[0] = 0

#building the embedding matrix
for word, index in vocabulary.items():
    if word in word2vec.vocab:
        embeddings[index] = word2vec.word_vec(word)

#get the maximum length of sentences
max_seq_length = max(file_read.question1.map(lambda x: len(x)).max(),
                     file_read.question2.map(lambda x: len(x)).max())


#shuffle the dataset
file_read = file_read.sample(frac=1)

#calculating the size of training, dev and test
train_size = int(file_read.shape[0] * 0.75)
test_size = int(file_read.shape[0] * 0.125)

#split the data into train, dev and test
train = file_read[ : train_size]
test = file_read[train_size : train_size + test_size]
dev = file_read[train_size + test_size :]

#split the inputs and outputs in raw data
X_train = {'left': train.question1, 'right': train.question2}
X_dev = {'left': dev.question1, 'right': dev.question2}
X_test = {'left': test.question1, 'right': test.question2}
Y_train = train.is_duplicate
Y_dev = dev.is_duplicate
Y_test = test.is_duplicate

#padding zeros
for dataset, side in itertools.product([X_train, X_dev], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)



# Model variables
n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 64
n_epoch = 25

def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

# The visible layer
# Input length is the normalized length of sentence
left_input = Input(shape=(max_seq_length,), dtype='int32')
right_input = Input(shape=(max_seq_length,), dtype='int32')

#input_size = (batch_size, sequence_length)
#output size = (batch_size, sequence_length, output_dim)
embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length, trainable=False)

# Embedded version of the inputs
# size = (bacth_size, max_seq_length, embedding_dim)
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

# Since this is a siamese network, both sides share the same LSTM
# Output space is n_hidden
shared_lstm = LSTM(n_hidden)

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

# Calculates the distance as defined by the MaLSTM model
malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

# Pack it all up into a model
malstm = Model([left_input, right_input], [malstm_distance])

# Adadelta optimizer, with gradient clipping by norm
optimizer = Adadelta(clipnorm=gradient_clipping_norm)

malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

# Start training
training_start_time = time()

malstm_trained = malstm.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, nb_epoch=n_epoch,
                            validation_data=([X_dev['left'], X_dev['right']], Y_dev))

print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))

# Plot accuracy
plt.plot(malstm_trained.history['acc'])
plt.plot(malstm_trained.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot loss
plt.plot(malstm_trained.history['loss'])
plt.plot(malstm_trained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()