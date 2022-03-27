import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder


#Data Preparation
with open('intents.json') as file:
	data = json.load(file)

trainingSentences = []
trainingLabels = []
labels = []
responses = []

for intent in data['intents']:
	for pattern in intent['patterns']:
		trainingSentences.append(pattern)
		trainingLabels.append(intent['tag'])
	responses.append(intent['responses'])

	if intent['tag'] not in labels:
		labels.append(intent['tag'])

numClasses = len(labels)


#Label Encoder
lblEncoder = LabelEncoder()
lblEncoder.fit(trainingLabels)
trainingLabels = lblEncoder.transform(trainingLabels)


#Tokenization
vocabSize = 1000
embeddingDim = 16
maxLength = 20
oovToken = "<OOV>"

tokenizer = Tokenizer(num_words=vocabSize, oov_token=oovToken)
tokenizer.fit_on_texts(trainingSentences)
wordIndex = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(trainingSentences)
paddedSequences = pad_sequences(sequences, truncating='post', maxlen=maxLength)


#Training A Neural Network
model = Sequential()
model.add(Embedding(vocabSize, embeddingDim, input_length=maxLength))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(numClasses, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.summary()
epochs = 500
history = model.fit(paddedSequences, np.array(trainingLabels), epochs=epochs)


#Saving the neural network
#Saving the trained model
model.save("chatModel")

#Saving the fitted tokenizer
import pickle
with open('tokenizer.pickle', 'wb') as handle:
	pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Saving the fitted label encoder
with open('labelEncoder.pickle', 'wb') as ecnFile:
	pickle.dump(lblEncoder, ecnFile, protocol=pickle.HIGHEST_PROTOCOL)

	
