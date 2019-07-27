

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field,BucketIterator
import pickle
from stanfordcorenlp import StanfordCoreNLP
from Configs import configs

config = configs()

def get_data():
    def tokenize_de(text):
        return nlp_de.word_tokenize(text)
    def tokenize_en(text):
        return nlp_en.word_tokenize(text)

    nlp_en = StanfordCoreNLP(r'E:\Learnings\NLP\corenlp', lang='en')
    nlp_de = StanfordCoreNLP(r'E:\Learnings\NLP\corenlp', lang='de')

    German = Field(tokenize=tokenize_de,init_token='<sos>',eos_token='<eos>',lower=True)
    English = Field(tokenize=tokenize_en,init_token='<sos>',eos_token='<eos>',lower=True)

    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(German, English))
    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validation examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")

    German.build_vocab(train_data, min_freq=2)
    English.build_vocab(train_data, min_freq=2)

    print(f"Unique tokens in source (de) vocabulary: {len(German.vocab)}")
    print(f"Unique tokens in target (en) vocabulary: {len(English.vocab)}")

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=config.BATCH_SIZE,
        device=config.DEVICE)
    return train_iterator, valid_iterator, test_iterator, German.vocab, English.vocab

if __name__ == "__main__":
    train_data ,_,_,_,_ = get_data()
    print(vars(train_data.examples[0]))






