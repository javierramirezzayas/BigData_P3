import json
import numpy as np
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
import requests
import os
import glob

# we're still going to use a Tokenizer here, but we don't need to fit it
tokenizer = Tokenizer(num_words=10000)

# for human-friendly printing
labels = ['does_not_talk_about_med_cond', 'does_talk_about_med_cond', 'ambiguous']

path = 'spark_tweets/'
extension = 'csv'
os.chdir(path)
csv_path = glob.glob('*.{}'.format(extension))
path = path + csv_path[0]
path = str(path)
os.chdir('..')

data = np.genfromtxt(path, delimiter=',', usecols=(0), dtype=None, encoding='utf-8')

evalSentenceArr = np.asarray(data)

# read in our saved dictionary
with open('training_models/dictionary.json', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)


# this utility makes sure that all the words in your input
# are registered in the dictionary
# before trying to turn them into a matrix.
def convert_text_to_index_array(text):
    words = kpt.text_to_word_sequence(text)
    wordIndices = []
    for word in words:
        if word in dictionary:
            wordIndices.append(dictionary[word])
        else:
            # print("'%s' not in training corpus; ignoring." %(word))
            pass
    return wordIndices


def evaluateSentence(evalSentenceArr):
    count = 0
    count_pos = 0
    count_neg = 0
    count_amb = 0

    for evalSentence in evalSentenceArr:
        testArr = convert_text_to_index_array(evalSentence)
        # print(testArr)
        input = tokenizer.sequences_to_matrix([testArr], mode='binary')
        # predict which bucket your input belongs in
        pred = model.predict(input)
        if labels[np.argmax(pred)] == 'does_not_talk_about_med_cond':
            count_neg = count_neg + 1
        elif labels[np.argmax(pred)] == 'does_talk_about_med_cond':
            count_pos = count_pos + 1
        elif labels[np.argmax(pred)] == 'ambiguous':
            count_amb = count_amb + 1
        # print("%s sentiment; %f%% confidence" % (labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))
        # count = count + 1
        # if count == 10:
        #     break
    print("Using Model: ")
    print("     does_talk_about_med_cond: " + str(count_pos))
    print("     does_not_talk_about_med_cond: " + str(count_neg))
    print("     ambiguous: " + str(count_amb))
    return [count_pos, count_neg, count_amb]

# read in your saved model structure
json_file = open('training_models/models_results.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
# and create a model from that
model = model_from_json(loaded_model_json)
# and weight your nodes with your saved values
model.load_weights('training_models/model1_weights.h5')

label_counts_model1 = evaluateSentence(evalSentenceArr)

# read in your saved model structure
json_file = open('training_models/model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
# and create a model from that
model = model_from_json(loaded_model_json)
# and weight your nodes with your saved values
model.load_weights('training_models/model2_weights.h5')

label_counts_model2 = evaluateSentence(evalSentenceArr)


# Post to dashboard
url = 'http://localhost:5001/updateData'
request_data = {'label_model1': str(labels), 'data_model1': str(label_counts_model1), 'label_model2': str(labels), 'data_model2': str(label_counts_model2)}
with open('graph_values/models_results.json', 'w') as outfile:
    json.dump(request_data, outfile)
    outfile.close()
requests.post(url, data=request_data)
