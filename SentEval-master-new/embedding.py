# Elmo 
import tensorflow_hub as hub
import tensorflow as tf 

# GPT2 
import numpy as np 
import torch
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

from bert_serving.client import BertClient
from gensen.gensen import GenSen, GenSenSingle

#ELMO
elmo = None
graph = None
sess = None
# Glove
embeddings_index = None

# GPT2
tokenizer = None
model = None 

# Bert
bc = None

# GPT2
tokenizer = None
model = None 

# Bert
bc = None

#Gensen

gensen_1 = None
gensen_2 = None

def getGensenEmb(batch,needPad = False,padLength = 30):
	global gensen_1
	global gensen_2

	if gensen_1 == None:
		gensen_1 = GenSenSingle(model_folder='gensen/data/models',filename_prefix='nli_large_bothskip',pretrained_emb='gensen/data/embedding/glove.840B.300d.h5')
		gensen_2 = GenSenSingle(model_folder = 'gensen/data/models',filename_prefix ='nli_large',pretrained_emb='./data/embedding/glove.840B.300d.h5')
	output_1, sent_1 = gensen_1.get_representation(batch, pool='last', return_numpy=True, tokenize=False)
	output_2, sent_2 = gensen_1.get_representation(batch, pool='last', return_numpy=True, tokenize=False)
	maxLength = 0 
	for i in range(len(output_1)):
		if len(output_1[i])> maxLength:
			maxLength = len(output_1[i])
	length = []
	output = []

	for i in range(len(output_1)):
		temp = np.concatenate((output_1[i],output_2[i]),axis = 1)
		length.append(len(temp[~np.all(temp==0,axis = 1)]))
		#print(length[-1])
		#if needPad:
		#	if needPad == -1:
		#		if len(temp) <= maxLength:
		#			temp = np.concatenate([temp,np.zeros([maxLength - len(temp),temp.shape[1]])])
		#		else:
		#			temp = temp[:maxLength]
		#	else:
		#		if len(temp) <= padLength:
		#			temp = np.concatenate([np.array(wordList),np.zeros([padLength - len(wordList),300])])
			#	else:
			#		temp = temp[:padLength]
		output.append(temp)  

	return np.array(output),np.array(length) 
def getGloveEmb(batch, needPad = False, padLength = 30, path = "glove.840B.300d.txt"):
	# Input format should be wordList (After tokenization)
	# padLength == -1, pad to the max length in this batch 
	global embeddings_index

	length = []
	output = []
	maxLength = 0 

	for sentence in batch:
		if len(sentence) > maxLength:
			maxLength  = len(sentence)

	if embeddings_index == None or len(embeddings_index) == 0:
		print("Load embedding")
		embeddings_index = dict()
		print(path)
		f = open(path  , 'r', errors = 'ignore', encoding='utf8')  # "word2vec.6B.50d.txt" for test 
		for line in f:
			splitLine = line.split(' ')  # For 3B  splitLine = line.split()
			word = splitLine[0]
			coefs = np.array([float(val) for val in splitLine[1:]])
			embeddings_index[word] = coefs
		f.close()
		print("Finish loading")

	for sentence in batch:
		#print(sentence)
		wordList = []
		for vocab in sentence:
			try:
				wordList.append(embeddings_index[vocab])
			except:
				wordList.append(np.zeros(300))
		length.append(len(wordList))

		if needPad:
			if padLength == -1:
				if len(sentence) < maxLength:
					wordList = np.concatenate([np.array(wordList),np.zeros([maxLength - len(wordList),300])])
			else:
				if len(wordList) < padLength:
					wordList = np.concatenate([np.array(wordList),np.zeros([padLength - len(wordList),300])])
				else:
					wordList = np.array(wordList)[:padLength]

		output.append(wordList)


	return np.array(output),np.array(length) 


def getElmoEmb(batch, needPad = False, padLength = 30):
	# Input format should be string without tokenization 
	assert type(batch[0]) is str
	global elmo 
	global graph 


	if elmo == None:
		graph = tf.Graph()
		with tf.Session(graph = graph) as sess:
			elmo = hub.Module("https://tfhub.dev/google/elmo/2")
			embeddings = elmo(batch,as_dict=True)["elmo"]
			sess.run(tf.global_variables_initializer())	
			output = embeddings.eval(session = sess)	
	else:
		with tf.Session(graph = graph) as sess:
			embeddings = elmo(batch,as_dict=True)["elmo"]
			sess.run(tf.global_variables_initializer())
			output = embeddings.eval(session = sess)

	if needPad:
		output_temp = []
		for i,sent in enumerate(output):
			if len(sent) > padLength:
				output_temp.append(sent[:padLength,:])
			else:
				z = np.zeros([padLength-sent.shape[0],sent.shape[1]])
				output_temp.append(np.concatenate([sent,z]))
		output = np.array(output_temp)

	return output, [len(output[0])]*len(batch) 

def getGPT2Emb(batch, needPad = False, padLength = 30):
	# Input format should be word List(After tokenization)
	#assert type(batch[0]) is not str
	output = []
	length = []

	global model
	global tokenizer
	if model == None:
		tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
		model = GPT2Model.from_pretrained('gpt2')

	for text in batch:
		indexed_tokens_1 = tokenizer.encode(text)
		tokens_tensor_1 = torch.tensor([indexed_tokens_1])
		with torch.no_grad():
			hidden_states_1, past = model(tokens_tensor_1)
			length.append(len(hidden_states_1[0]))
			output.append(hidden_states_1[0].numpy())

	if needPad:
		output_temp = [ ]
		maxLength = np.array(length).max()

		if padLength == -1 :
			for i,hidden_states_1 in enumerate(output):
				if hidden_states_1.shape[0] < maxLength:
					output_temp.append(np.concatenate([hidden_states_1,np.zeros([maxLength-hidden_states_1.shape[0],hidden_states_1.shape[1]])]))
				else:
					output_temp.append(hidden_states_1[:maxLength,:])
		else:
			for i,hidden_states_1 in enumerate(output):
				if hidden_states_1.shape[0] < padLength:
					output_temp.append(np.concatenate([hidden_states_1,np.zeros([padLength-hidden_states_1.shape[0],hidden_states_1.shape[1]])]))
				else:
					output_temp.append(hidden_states_1[:padLength,:])

		output = np.array(output_temp)

	return np.array(output),np.array(length)

def getBertEmb(batch,needPad = False, padLength = 30):
	global bc
	if bc == None:
		bc = BertClient()
	output = bc.encode(batch)
	length = []
	output1 = []

	for i,emb in enumerate(output):
		output1.append(emb[~np.all(emb == 0, axis=1)])
		#print("output_1:",len(output1[-1]))
		length.append(len(output1[i]))
	return np.array(output),np.array(length)


