import nltk
import feature_extract
import pandas as pd
import numpy as np

file = pd.read_csv("/Users/williamshelton/Desktop/rotten_tomatoes/train.tsv", sep="\t")
test = pd.read_csv("/Users/williamshelton/Desktop/rotten_tomatoes/test.tsv", sep="\t")

tupized = feature_extract.file_to_tuple_of_column(file, "Phrase")

classifier = nltk.NaiveBayesClassifier.train(tupized)

df = pd.DataFrame(np.zeros(0, dtype=[('PhraseId', 'i4'), ('Sentiment', 'i4')]))

for index, row in test.iterrows():
# 	print feature_extract.extract_features(row["Phrase"].split())
	sentiment = classifier.classify(feature_extract.extract_features(row["Phrase"].split()))
	print sentiment
	df = df.append({'PhraseId':row['PhraseId'], 'Sentiment':sentiment},ignore_index=True)

print classifier.show_most_informative_features(2)

df.to_csv("/Users/williamshelton/Desktop/rotten_tomatoes/results.csv")