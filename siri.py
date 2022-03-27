import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import colorama
import os
from colorama import Fore, Style
import pickle
import warnings
import speech_recognition as sr
import pyttsx3


warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
colorama.init()
r = sr.Recognizer()


with open('intents.json') as file:
	data = json.load(file)


model = keras.models.load_model('chatModel')

with open('tokenizer.pickle', 'rb') as handle:		
    tokenizer = pickle.load(handle)

with open('labelEncoder.pickle', 'rb') as enc:
    lblEncoder = pickle.load(enc)

maxLen = 20
print(Fore.YELLOW + "Start talking to the chatbot and say quit to exit" + Style.RESET_ALL)


while True:
    print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
    try: 
        with sr.Microphone() as source2:
            r.adjust_for_ambient_noise(source2, duration=0.001)           
            audio2 = r.listen(source2)
            inp = r.recognize_google(audio2)
            inp = inp.lower()
            
            print(inp)

            if inp == 'quit':
                break
            
            result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]), truncating='post', maxlen=maxLen))
            tag = lblEncoder.inverse_transform([np.argmax(result)])
            
            for i in data['intents']:
                if i['tag'] == tag:
                    output = np.random.choice(i['responses'])
                    print(Fore.GREEN + "Chatbot: " + Style.RESET_ALL, output)
                    engine = pyttsx3.init()
                    engine.say(output)
                    engine.runAndWait()

    except sr.RequestError as e:
        print(f"Could not request results: {e}")

    except Exception as e:
        print(e)