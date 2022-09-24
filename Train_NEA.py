#%%

#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import math
from datetime import datetime

import numpy as np
import scipy.io as sio
import time
from scipy import spatial
#from nltk.corpus import stopwords
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#%%
#import evaluation
#Function to generate a training batch for the nureal embedding allocation model
def generate_batch(batch_size, phi, theta, V): # phi=word topic distribution #theta=topic document distribution #V= # of words in a dictionary
   batch = np.ndarray(shape=(batch_size), dtype=np.int32)
   labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
   i = 0
   D = len(theta) #D=document #K=topic
   K = len(phi)
   while i<batch_size:
        d = int(random.uniform(0, len(theta)))
        z= np.random.choice(K, p=theta[d]) 
        v= np.random.choice(V, p=phi[z])
        batch[i] = z
        labels[i, 0] = v
        #print(d, z, v, D, D+z)
        i+=1
        #if d==0: # in [0,1]:
              #print(topicdocdictionary[d],z, "-->", dictionary[v])
   return batch, labels
#%%
# LDA Hyperparameters
#K=1000
alpha=0.1
beta=0.01
#NEA hyperparameters
num_steps=1000000
batch_size=128
embedding_size=300
num_sampled=5
docids=None
ns=True
#def fit(num_steps, batch_size, embedding_size, num_sampled, folder, docids, corpus, ns=True):
print(str(datetime.now()), "Training LDA model...")
print(str(datetime.now()), "Calculating statistics and estimators...")
#Load the Dictionary
f=open('data/NIPSdict.txt','r')
dictionary=f.read().split()
f.close
#%%
# Calculating theta and phi from LDA  
doc_topic_matrix=np.loadtxt('LDA-Parameters/LDATopicModel_docTopicCounts.txt')
word_topic_matrix=np.loadtxt('LDA-Parameters/LDATopicModel_wordTopicCounts.txt') 
theta = np.asarray(doc_topic_matrix, dtype=np.float)
phi = np.asarray(word_topic_matrix.T, dtype=np.float)
theta = theta + alpha
theta_norm = np.sum(theta, axis=1)[:, np.newaxis]
theta = theta / theta_norm
phi = phi + beta
phi_norm = np.sum(phi, axis=1)[:, np.newaxis]
phi = phi / phi_norm
#%%
print(str(datetime.now()), "Training Neural Embedding Allocation...")
D = len(theta)
print('Number of Documents: ', len(theta))
K = len(phi)
print('Number of topics: ',len(phi))
V=len(dictionary) # vocabulary size
print('Dictionary size: ', len(dictionary))
#%%
# Training NEA
graph = tf.Graph()
with graph.as_default():
   train_inputs = tf.placeholder(tf.int32, shape=[batch_size]) #input data
   train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
   with tf.device('/cpu:0'): # Ops and variables pinned to the CPU because of missing GPU implementation
      embeddings = tf.Variable(tf.random_uniform([K, embedding_size], -1.0, 1.0)) #Look up embeddings for inputs.
      embed = tf.nn.embedding_lookup(embeddings, train_inputs)
# Construct the variables for the NCE loss
      nce_weights = tf.Variable(tf.truncated_normal([V, embedding_size],
                    stddev=1.0 / math.sqrt(embedding_size)))
      nce_biases = tf.Variable(tf.zeros([V]))
   if not ns:
       loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=embed,
                   num_sampled=num_sampled, num_classes=V))
   else:
       loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=embed,
                   num_sampled=num_sampled, num_classes=V))
   # Compute the average NCE loss for the batch.tf.nce_loss automatically draws a new sample of the negative labels each time we evaluate the loss.
   #loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, inputs=embed, labels=train_labels, num_sampled=num_sampled, num_classes=V))
   # Construct the SGD optimizer
   optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
   # Compute the cosine similarity between minibatch examples and all embeddings.
   norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
   normalized_embeddings = embeddings / norm
   # Add variable initializer.
   init = tf.global_variables_initializer()
#Begin training
start = time.time()
with tf.Session(graph=graph) as session:
# We must initialize all variables before we use them.
   init.run()
   average_loss = 0
   for step in xrange(num_steps):
      batch_inputs, batch_labels = generate_batch(
batch_size=batch_size, phi=phi, theta=theta, V=V)
      feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

   # We perform one update step by evaluating the optimizer op (including it in the list of returned values for session.run()
      _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
   if step % 10000 == 0:
          if step > 0:
              average_loss /= 10000
          print('Average loss at step ', step, ': ', average_loss)

   runtime_nea = time.time() - start
   norms=norm.eval()
   final_embeddings = normalized_embeddings.eval()
   embeddings_ = embeddings.eval()
   nce_weights_ = nce_weights.eval()
   nce_biases_ = nce_biases.eval()
   soft_int=np.dot(final_embeddings,nce_weights_.T)
   softExp = np.exp(soft_int)
   NEA_Topics=np.apply_along_axis(lambda x: x/x.sum(),1,softExp)

#%% Evaluate NEA
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

# Load corpus
doc_word_matrix=list()
filename='data/NIPS.txt'
f=open(filename,'r')
for line in f:
    doc_word_matrixTemp=list()
    tmp=line.split()
    for word in tmp:
        doc_word_matrixTemp.append(int(word)) 
    doc_word_matrix.append(doc_word_matrixTemp)
f.close()

Corpus = []
for i in range(len(doc_word_matrix)):
    doc=[]
    for j in doc_word_matrix[i]:
        doc.append(dictionary[j])
    Corpus.append(doc)
f.close()

dct = Dictionary(Corpus)

corpus_bow = [dct.doc2bow(text) for text in Corpus]

numToget=10
nea_topics_list=[]
for m in range(len(NEA_Topics)):
    temp=NEA_Topics[m,]
    tmpIndx=[i[0] for i in sorted(enumerate(temp),key=lambda x:x[1],reverse=True)]
    topWordTmp=[]
    for n in range(numToget):
        topWordTmp.append(dictionary[tmpIndx[n]])
        nea_topics_list.append(topWordTmp)
cm = CoherenceModel(topics=nea_topics_list, corpus=corpus_bow, dictionary=dct, coherence='u_mass')
coherence_nea = cm.get_coherence() 

print('UMASS coherence of NEA reconstructed topics', coherence_nea)

# Find top words 
numToget=10
topProb=[]
topWord=[]
for m in range(len(NEA_Topics)):
    temp=NEA_Topics[m,]
    tmpIndx=[i[0] for i in sorted(enumerate(temp),key=lambda x:x[1],reverse=True)]
    topProbTmp=[]
    topWordTmp=[]
    for n in range(numToget):
        topProbTmp.append(temp[tmpIndx[n]])
        topWordTmp.append(dictionary[tmpIndx[n]])
    topProb.append(topProbTmp)
    topWord.append(topWordTmp)

print('Top 10 words from NEA reconstructed topics: \n',topWord)