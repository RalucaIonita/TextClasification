# import stuff
import numpy as np
import re
import os
import os
from collections import defaultdict
from sklearn import svm

words_number = 2000


# accuracy
def accuracy(predicted, labels):
	return 100 * (predicted == labels).astype('int').mean()


# function for searching through a folder
def files_in_folder(path):
	files = []
	for file in os.listdir(path):
		if os.path.isfile(os.path.join(path, file)):
			files.append(os.path.join(path, file))
	return sorted(files)

# function for eliminating the extension
def file_without_extension(path):
	file_name = os.path.basename(path)
	file_name_without_extension = file_name.replace('.txt', '')
	return file_name_without_extension


# function for reading from folders
def read_texts(path):
	text_data = []
	text_index = []
	for file in files_in_folder(path):
		file_index = file_without_extension(file)
		text_index.append(file_index)
		with open(file, 'r', encoding = 'utf-8') as file_in:
			text = file_in.read()
#			text = text.lower() # to lower case
		text_without_punctuation = re.sub("[.,-:;!?\"\'\/()_*=`]", "", text)
		words = text_without_punctuation.split()
		text_data.append(words)
	return (text_index, text_data)


# paths
data_path = 'data/'
training_data_path = data_path + 'train/'
labels_path = data_path + 'labels_train.txt'
test_data_path = data_path + 'test/'


# reading the data
labels = np.loadtxt(labels_path)
training_index, train = read_texts(training_data_path)
testing_index, test = read_texts(test_data_path)

data = train + test

print('Train data: ', len(train))
print('Test data: ', len(test))
print('Data:', len(data))


# counting the words
word_counter = defaultdict(int)
for text in data:
	for word in text:
		word_counter[word] += 1


# turning the dictionary into a touple [word, frequency]
word_touples = list(word_counter.items())


# ordering the touple list
word_touples = sorted(word_touples, key = lambda kv: kv[1], reverse = True)


# extracting the first n (words_number) most frequent words
word_touples = word_touples[0:words_number]

print('First 10 most frequent words: ', word_touples[0:10])


# selecting the words
list_of_selected_words = []
for word, frequency in word_touples:
	list_of_selected_words.append(word)


# bag of words
def get_bow(text, list_of_words):
	counter = dict()
	words = set(list_of_words)
	for word in words:
		counter[word] = 0
	for word in text:
		if word in words:
			counter[word] += 1
	return counter

def get_bow_on_data(data, list_of_words):
	bow = np.zeros((len(data), len(list_of_words)))
	for index, text in enumerate(data):
		bow_dictionary = get_bow(text, list_of_words)
		# normalize
		v = np.array(list(bow_dictionary.values()))
		v = v / np.sqrt(np.sum(v ** 2))
		bow[index] = v
	return bow

data_bow = get_bow_on_data(data, list_of_selected_words)
print('Data bow shape: ', data_bow.shape)


# data size
train_data_size = 2450
validation_data_size = len(train) - train_data_size
test_data_size = len(test)
data_size = len(data)

print('Train data size: ', train_data_size)
print('Validation data size: ', validation_data_size)
print('Test data size: ', test_data_size)
print('Data size: ', data_size)

# data indexes
train_indexes = np.arange(0, train_data_size)
validation_indexes = np.arange(train_data_size, train_data_size + validation_data_size)
test_indexes = np.arange(train_data_size + validation_data_size + 1, data_size+1)

print(test_indexes)

# SVM
#for C in [0.001, 0.01, 0.1, 1, 10, 100]:
clasifier = svm.SVC(C = 100, kernel = 'linear')
clasifier.fit(data_bow[train_indexes, :], labels[train_indexes])
predictions = clasifier.predict(data_bow[validation_indexes, :])
print("Validation accuracy with C = 100 : ", accuracy(predictions, labels[validation_indexes]))



test_bow = get_bow_on_data(test, list_of_selected_words)
print('Test bow shape', test_bow.shape)

predicted_labels = clasifier.predict(test_bow)
'''
for index, label in zip(test_indexes, predicted_labels):
	print(index, label)

file = "submisie_Kaggle_1.csv"
np.savetxt(file, np.stack((test_indexes, predicted_labels)).T, fmt = "%d", delimiter=',', header = "Id,Prediction", comments = '')
'''
