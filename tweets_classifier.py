import numpy as np
import json
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

# extract_data from csv
training = np.genfromtxt('cleantextlabels7.csv', delimiter=',', usecols=(0, 1), dtype=None, encoding='utf-8')

# create training data from the tweets
train_x = [x[1] for x in training]

# Index all the sentiment labels
train_y = np.asarray([x[0] for x in training])

# Work with 10000 most popular words found in dataset
max_words = 10000

# Create a new Tokenizer
tokenizer = Tokenizer(num_words=max_words)

# Feed our tweets to the Tokenizer
tokenizer.fit_on_texts(train_y)

dictionary = tokenizer.word_index

# Let's save this out so we can use it later
with open('training_models/dictionary.json', 'w') as dictionary_file:
    json.dump(dictionary, dictionary_file)


def convert_text_to_index_array(text):
    return [dictionary[word] for word in kpt.text_to_word_sequence(text)]


allWordIndices = []

# For each tweet, change each token to its ID in the Tokenizer's word_index
for text in train_y:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)

# List of all tweets converted to index arrays. Cast as an array for future usage.
allWordIndices = np.asarray(allWordIndices)

# Create one-hot matrices out of the indexed tweets
train_y = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')

# Treat labels as categories (There are three categories)
train_x = keras.utils.to_categorical(train_x, 3)


# FOR MODEL 1

# Initialize simple neural net
model1 = Sequential()
# Standard Linear neural net layer. Input sentence that will be turned into matrix of max_words. 512 Outputs
model1.add(Dense(512, input_shape=(max_words, ), activation='relu'))
# Randomly Remove Data
model1.add(Dropout(0.5))
model1.add(Dense(256, activation='sigmoid'))
model1.add(Dropout(0.5))
model1.add(Dense(3, activation='softmax'))

# Compile the network
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train Network
model1.fit(train_y, train_x, batch_size=32, epochs=3, verbose=1, validation_split=0.2, shuffle=True)

# Save the model
model1.save_weights('training_models/model1_weights.h5')

model1_json = model1.to_json()
with open('training_models/models_results.json', 'w') as json_file:
    json_file.write(model1_json)


# FOR MODEL 2

# Initialize simple neural net
model2 = Sequential()
# Standard Linear neural net layer. Input sentence that will be turned into matrix of max_words. 512 Outputs
model2.add(Dense(1024, input_shape=(max_words, ), activation='sigmoid'))
# Randomly Remove Data
model2.add(Dropout(0.5))
model2.add(Dense(3, activation='sigmoid'))

# Compile the network
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train Network
model2.fit(train_y, train_x, batch_size=32, epochs=3, verbose=1, validation_split=0.2, shuffle=True)

# Save the model
model2.save_weights('training_models/model2_weights.h5')

model2_json = model2.to_json()
with open('training_models/model2.json', 'w') as json_file:
    json_file.write(model2_json)
