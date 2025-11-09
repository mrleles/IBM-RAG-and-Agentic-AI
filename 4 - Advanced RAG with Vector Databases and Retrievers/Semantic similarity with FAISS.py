'''
!pip install faiss-cpu numpy scikit-learn
!pip install "tensorflow>=2.0.0"
!pip install --upgrade tensorflow-hub
'''

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import faiss
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

# Suppressing warnings
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')

newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

def preprocess_text(text):
    text = re.sub(r'^From:.*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S*@\S?', '', text)