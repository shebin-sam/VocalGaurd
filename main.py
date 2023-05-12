from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import speech_recognition as sr
import pyttsx3
from_disk = pickle.load(open("toxicvect.pkl", "rb"))
vectorizer = TextVectorization.from_config(from_disk['config'])
# You have to call `adapt` with some dummy data (BUG in Keras)
vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
vectorizer.set_weights(from_disk['weights'])

model = tf.keras.models.load_model('toxic.h5')


text_speech = pyttsx3.init()

# Initialize recognizer
r = sr.Recognizer()
while True:
# Use the microphone as the audio source
    with sr.Microphone() as source:
        print("Speak now...")
        text_speech.say("Speak Now ..")
        text_speech.runAndWait()
        audio = r.listen(source)

    try:
    # Recognize speech using Google Speech Recognition
        text = r.recognize_google(audio)
        print("You said: {}".format(text))
        b = vectorizer([text])
        c =model.predict(b)
        d=[]
        for i in c[0]:
            if i > 0.5 :
                print(1)
                d.append(1)
            else:
                print(0)
                d.append(0)

        print(d)
        ans =''
        e = ['toxic','extremly toxic','obscene','threatening','insulting','racial']
        for i in range(6):
            if d[i] ==1:
                ans += e[i] +' '

        if 1 in d:
            ans = ans + 'speech'
            print(ans)
            text_speech.say(ans)
            text_speech.runAndWait()
        else:
            print("normal speech")
            text_speech.say("Normal Speech")
            text_speech.runAndWait()
    except sr.UnknownValueError:
        print("Oops! Could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    