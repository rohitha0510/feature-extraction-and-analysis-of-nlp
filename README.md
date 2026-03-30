# feature-extraction-and-analysis-of-nlp
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np

import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.layers import Bidirectional,GRU
from keras.models import model_from_json
import pickle

main = tkinter.Tk()
main.title("Feature Extraction and Analysis of Natural Language Processing for Deep Learning English Language") 
main.geometry("1300x1200")

global filename
global classifier
global char_to_int
global int_to_char
vocab_list = []
dataX = []
dataY = []
global n_vocab

def upload(): 
    global filename
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def getID(chars,data):
    index = 0
    for i in range(len(chars)):
        if chars[i] == data:
            index = i;
            break
    return index       

def preprocess():
    global n_vocab
    dataX.clear()
    dataY.clear()
    global char_to_int
    global int_to_char
    global filename
    text.delete('1.0', END)
    sentences = ''
    with open(filename, "r") as file:
        for line in file:
            line = line.strip('\n')
            line = line.strip()
            line.lower()
            sentences+=line+" "
    file.close()
    sentences = sentences.strip()
    vocab_list.clear()
    arr = sentences.split(" ")
    for i in range(len(arr)):
        vocab_list.append(arr[i])

    raw_text = sentences.lower()
    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    n_chars = len(raw_text)
    n_vocab = len(chars)
    text.insert(END,"Text Preprocessing Completed\n")
    text.insert(END,"Total Characters: "+str(n_chars)+"\n")
    text.insert(END,"Total Vocab: "+str(n_vocab)+"\n")
    for i in range(0, n_chars):    
        dataX.append(char_to_int.get(raw_text[i]))
        dataY.append(getID(chars,raw_text[i]))       


def runBILSTM():
    global n_vocab
    global classifier
    text.delete('1.0', END)
    n_patterns = len(dataX)
    if os.path.exists('model/lstmmodel.json'):
        with open('model/lstmmodel.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        json_file.close()
        classifier.load_weights("model/lstmmodel_weights.h5")
        classifier._make_predict_function()
        f = open('model/lstmhistory.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        loss = data['loss']
        loss = loss[9]
        text.insert(END,"BI-LSTM Training Model Loss = "+str(loss)+"\n")
    else:
        seq_length = 1
        X = np.reshape(dataX, (n_patterns, seq_length, 1))
        X = X / float(n_vocab)
        y = np_utils.to_categorical(dataY)
        print(X.shape)
        print(y.shape)
        model = Sequential()
        model.add(Bidirectional(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(256)))
        model.add(Dropout(0.2))
        model.add(Dense(y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        hist = model.fit(X, y, epochs=800, batch_size=64)
        model.save_weights('model/lstmmodel_weights.h5')            
        model_json = model.to_json()
        with open("model/lstmmodel.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/lstmhistory.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        loss = hist.history['loss']
        loss = loss[9]
        text.insert(END,"BI-LSTM Training Model Loss = "+str(loss)+"\n")
        classifier = model
        
def runGRU():
    global n_vocab
    global classifier
    n_patterns = len(dataX)
    print(os.path.exists('model/grumodel.json'))
    if os.path.exists('model/grumodel.json'):
        with open('model/grumodel.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        json_file.close()
        classifier.load_weights("model/grumodel_weights.h5")
        classifier._make_predict_function()
        f = open('model/gruhistory.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        loss = data['loss']
        loss = loss[9]
        text.insert(END,"\nBI-GRU Training Model Loss = "+str(loss)+"\n")
    else:
        seq_length = 1
        X = np.reshape(dataX, (n_patterns, seq_length, 1))
        X = X / float(n_vocab)
        y = np_utils.to_categorical(dataY)
        print(X.shape)
        print(y.shape)
        model = Sequential()
        model.add(Bidirectional(GRU(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(GRU(256)))
        model.add(Dropout(0.2))
        model.add(Dense(y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        hist = model.fit(X, y, epochs=200, batch_size=64)
        model.save_weights('model/grumodel_weights.h5')            
        model_json = model.to_json()
        with open("model/grumodel.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/gruhistory.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        loss = hist.history['loss']
        loss = loss[9]
        text.insert(END,"\nBI-GRU Training Model Loss = "+str(loss)+"\n")
        classifier = model
    

def graph():
    f = open('model/lstmhistory.pckl', 'rb')
    lstm = pickle.load(f)
    lstm = lstm['loss']
    f.close()
    f = open('model/gruhistory.pckl', 'rb')
    gru = pickle.load(f)
    gru = gru['loss']
    f.close()

    lstm_arr = []
    gru_arr = []
    for i in range(0,100):
        lstm_arr.append(lstm[i])
        gru_arr.append(gru[i])

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Epoch/Iterations')
    plt.ylabel('Loss')
    plt.plot(lstm_arr, 'ro-', color = 'blue')
    plt.plot(gru_arr, 'ro-', color = 'orange')
    plt.legend(['BI-LSTM Loss', 'BI-GRU Loss'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title('BI-LSTM vs BI-GRU Loss Graph')
    plt.show()        

def predict():
    print(vocab_list)
    text.delete('1.0', END)
    testfile = filedialog.askopenfilename(initialdir="Dataset")
    with open(testfile, "r") as file:
        for line in file:
            line = line.strip('\n')
            line = line.strip()
            line.lower()
            output = ''
            segment = ''
            for i in range(len(line)):
                data = char_to_int[line[i]]
                temp = []
                temp.append(data)
                temp = np.asarray(temp)
                x = np.reshape(temp, (1, temp.shape[0], 1))
                x = x / float(n_vocab)
                prediction = classifier.predict(x, verbose=0)[0]
                index = np.argmax(prediction)
                result = int_to_char[index]
                output+=result
                print(str(result)+" "+str(data)+" "+str(output in vocab_list))
                if output in vocab_list and output != 'for' and output != 'form' and output != 'be' and len(output) > 2:
                    segment+=output+" "
                    output = ''
            text.insert(END,"Input Sentence : "+str(line)+"\n")
            text.insert(END,"Ouput Word Segmentation : "+segment+"\n\n")
    
font = ('times', 16, 'bold')
title = Label(main, text='Feature Extraction and Analysis of Natural Language Processing for Deep Learning English Language')
title.config(bg='firebrick4', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload WIKI Sentence Dataset", command=upload, bg='#ffb3fe')
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=preprocess, bg='#ffb3fe')
processButton.place(x=440,y=550)
processButton.config(font=font1) 

lstmButton1 = Button(main, text="Generate Word Segmentation BI-LSTM Model", command=runBILSTM, bg='#ffb3fe')
lstmButton1.place(x=670,y=550)
lstmButton1.config(font=font1) 

gruButton = Button(main, text="Generate Word Segmentation BI-GRU Model", command=runGRU, bg='#ffb3fe')
gruButton.place(x=50,y=600)
gruButton.config(font=font1) 

graphButton = Button(main, text="Loss Comparison Graph", command=graph, bg='#ffb3fe')
graphButton.place(x=440,y=600)
graphButton.config(font=font1) 

predictButton = Button(main, text="Word Segmentation Prediction", command=predict, bg='#ffb3fe')
predictButton.place(x=670,y=600)
predictButton.config(font=font1) 

main.config(bg='LightSalmon3')
main.mainloop()
