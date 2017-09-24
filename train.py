import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import pickle as cPickle
import os
import re
import collections
from tqdm import tqdm
import json

# load the user configs
logging.debug("Load the user configs")
with open('./conf/config.json') as f:
	config = json.load(f)

# config variables
logging.debug("Load config variables")
data_path = config["data_path"]
random_seed = config["seed"]
model = config["model"]
model_path = config["model_path"]
test_size = config["test_size"]

doc=[]
classes=[]
if not os.path.exists(model_path):
    os.mkdir(model_path)

logging.debug("Load Files from the folder")
for root, directories, filenames in os.walk(data_path):
     for file in filenames:
            if file != ".DS_Store":
                file_name, file_extension = os.path.splitext(file)
                classes.append(root.split("/")[-1])
                file1 = open("{}/{}".format(root,file),"r",encoding='utf-8', errors='ignore')
                doc.append(file1.read())

logging.debug("Creating DataFrame")
df=pd.DataFrame()
df['doc']=doc
df['classes']=classes


y=pd.get_dummies(df['classes'])
label_dict={}
for i in range(y.shape[1]):
    label_dict[i]=y.columns[i]
print("[INFO] saving label dictionary...")
logging.debug("saving label dictionary...")
joblib.dump(label_dict, "{}{}_label.pkl".format(model_path,model))
y =np.array(y)

logging.debug("Load SnowBalls Stemmer")
stemmer = SnowballStemmer('english').stem
def stem_tokenize(text):
     return [stemmer(i) for i in word_tokenize(text)]

logging.debug("Create Tokenizer")
vectorizer = CountVectorizer(analyzer='word',lowercase=True,tokenizer=stem_tokenize)
X = vectorizer.fit_transform(df.doc.values)
logging.debug("Saving Tokenizer")
joblib.dump(vectorizer, '{}{}_vectorizer.pkl'.format(model_path,model))

logging.debug("Creating classifier")
clf = MultinomialNB()
y = np.argmax(y, axis=1)
logging.debug("Training classifier")
clf.fit(X, y)

# dump classifier to file
print("[INFO] saving model...")
logging.debug("Saving model/classifier")
joblib.dump(clf, "{}{}_clf.pkl".format(model_path,model))
