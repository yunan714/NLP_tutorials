from torchtext import data
from torchtext.datasets import Multi30k
from torchtext.data import Field,BucketIterator,Dataset
from stanfordcorenlp import StanfordCoreNLP
from Configs import configs
import dill

config = configs()

nlp_en = StanfordCoreNLP(r'E:\Learnings\NLP\corenlp', lang='en')
nlp_de = StanfordCoreNLP(r'E:\Learnings\NLP\corenlp', lang='de')
def tokenize_de(text):
    return nlp_de.word_tokenize(text)
def tokenize_en(text):
    return nlp_en.word_tokenize(text)

German = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
English = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

def processed_data():
    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(German, English))
    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")
    dill.dump(train_data.examples,open('processed_data/train.bin','wb'))
    dill.dump(valid_data.examples,open('processed_data/valid.bin','wb'))
    dill.dump(test_data.examples,open('processed_data/test.bin','wb'))

    German.build_vocab(train_data, min_freq=2)
    English.build_vocab(train_data, min_freq=2)

    print(f"Unique tokens in source (de) vocabulary: {len(German.vocab)}")
    print(f"Unique tokens in target (en) vocabulary: {len(English.vocab)}")
    dill.dump(German.vocab,open('processed_data/Ge_vocab.bin','wb'))
    dill.dump(English.vocab,open('processed_data/En_vocab.bin','wb'))

class Multi30k_dataset(Dataset):
    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src),len(ex.trg))
    ## 一种方法可以是key相近的对应的映射也相近  不清楚其具体意思

def get_data():
    with open("processed_data/train.bin",'rb') as f:
        train_data_ex = dill.load(f)
    with open("processed_data/valid.bin",'rb') as f:
        valid_data_ex = dill.load(f)
    with open("processed_data/test.bin",'rb') as f:
        test_data_ex = dill.load(f)
    with open("processed_data/Ge_vocab.bin",'rb') as f:
        Ge_vocab = dill.load(f)
    with open("processed_data/En_vocab.bin", 'rb') as f:
        En_vocab = dill.load(f)

    fields = {'src':English,'trg':German}
    train_data = Multi30k_dataset(examples=train_data_ex, fields=fields)
    valid_data = Multi30k_dataset(examples=valid_data_ex, fields=fields)
    test_data = Multi30k_dataset(examples=test_data_ex, fields=fields)
    train_data.fields['src'].vocab = En_vocab
    train_data.fields['trg'].vocab = Ge_vocab
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=config.BATCH_SIZE,
        device=config.DEVICE)  #数据已经转为cuda了
    return train_iterator, valid_iterator, test_iterator, Ge_vocab, En_vocab

if __name__ == "__main__":
    train_iterator ,_,_,_,_ = get_data()






