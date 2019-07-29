import dill
from torchtext import data
from torchtext.datasets import Multi30k
from torchtext.data import Field,BucketIterator ,Dataset
from stanfordcorenlp import StanfordCoreNLP
from configs import configs
import os


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
    train_data, dev_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(German, English))
    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(dev_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")
    dill.dump(train_data.examples, open('processed_data/train.bin', 'wb'))
    dill.dump(dev_data.examples, open('processed_data/dev.bin', 'wb'))
    dill.dump(test_data.examples, open('processed_data/test.bin', 'wb'))

    German.build_vocab(train_data, min_freq=2)
    English.build_vocab(train_data, min_freq=2)

    print(f"Unique tokens in source (de) vocabulary: {len(German.vocab)}")
    print(f"Unique tokens in target (en) vocabulary: {len(English.vocab)}")
    dill.dump(German.vocab, open('processed_data/Ge_vocab.bin', 'wb'))
    dill.dump(English.vocab, open('processed_data/En_vocab.bin', 'wb'))

class Multi30k_dataset(Dataset):
    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src),len(ex.trg))
    ## 一种方法可以是key相近的对应的映射也相近  不清楚其具体意思

def get_data():
    train_ex = dill.load(open('processed_data/train.bin', 'rb'))
    dev_ex = dill.load(open('processed_data/dev.bin', 'rb'))
    test_ex = dill.load(open('processed_data/test.bin', 'rb'))
    fields = {'src':English,'trg':German}

    train_data = Multi30k_dataset(examples=train_ex,fields=fields)
    dev_data = Multi30k_dataset(examples=dev_ex, fields=fields)
    test_data = Multi30k_dataset(examples=test_ex, fields=fields)
    train_data.fields['src'].vocab = English
    train_data.fields['trg'].vocab = German

    train_iter, dev_iter, test_iter = BucketIterator.splits((train_data,dev_data,test_data),
                                                            batch_size=config.BATCH_SIZE,
                                                            device = config.DEVICE)
    return train_iter, dev_iter, test_iter, English.vocab, German.vocab

if __name__ == "__main__":
    if os.path.exists(config.DATA_PATH):
        t,_,_,_,_ = get_data()
        print(vars(t.examples[0]))
    else:
        os.mkdir(config.DATA_PATH)
        processed_data()