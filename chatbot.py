import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import colorama
import os
from colorama import Fore, Style
import pickle
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
colorama.init()
with open('intents.json') as file:
	data = json.load(file)

def chat():
	#Load trained mode
	model = keras.models.load_model('chatModel')

	#Load tokenizer object
	with open('tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)

	#Load label encoder object
	with open('labelEncoder.pickle', 'rb') as enc:
		lblEncoder = pickle.load(enc)

	#Parameters
	maxLen = 20

	while True:
		print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
		inp = input()
		if inp.lower() == 'quit':
			break

		result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]), 
		truncating='post', maxlen=maxLen))
		tag = lblEncoder.inverse_transform([np.argmax(result)])

		for i in data['intents']:
			if i['tag'] == tag:
				print(Fore.GREEN + "Chatbot: " + Style.RESET_ALL, np.random.choice(i['responses']))


print(Fore.YELLOW + "Start talking to the chatbot and enter quit to exit" + Style.RESET_ALL)
chat()