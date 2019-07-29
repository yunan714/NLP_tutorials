import dill
import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field,BucketIterator ,Dataset
from stanfordcorenlp import StanfordCoreNLP
from configs import configs

config = configs()

nlp_en = StanfordCoreNLP(r'E:\Learnings\NLP\corenlp', lang='en')
nlp_de = StanfordCoreNLP(r'E:\Learnings\NLP\corenlp', lang='de')
def tokenize_de(text):
    return nlp_de.word_tokenize(text)
def tokenize_en(text):
    return nlp_en.word_tokenize(text)

German = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
English = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

