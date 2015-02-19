import pandas as pd
import nltk as nltk

file = pd.read_csv("/Users/williamshelton/Desktop/rotten_tomatoes/train.tsv", sep="\t")

classifier = nltk.NaiveBayesClassifier.train(file);

print nltk.classify.accuracy(classifier, file);

# print file

# file['word_array'] = 'e'

# print file
# 
# for index, row in file.iterrows():
# 	file['word_array'][index] = row['Phrase'].split()
# 	print file['word_array'][index]