word_features = [
	'this',
	'love',
	'amazing',
	'horrible',
	'awful',
	'good'
]

def file_to_tuple_of_column(file, column_name):
	sentences = []
	for index, row in file.iterrows():
		t = (extract_features(row["Phrase"].split()), row["Sentiment"])
		sentences.append(t)
	return tuple(sentences)

def extract_features(document):
	document_words = set(document)
	features = {}
	for word in word_features:
		features['contains(%s)' % word] = (word in document_words)
	return features