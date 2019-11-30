import tarfile
from zipfile import ZipFile
import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from keras.preprocessing.sequence import pad_sequences
import json

class PreProcessor:
    def __init__(self):
        self.words_seen = set()
        self.tags_seen = set()

        self.files_seen = [] #keeps track of doc id's
        
        self.max_len = 0

    def extract_file(self,file_path:str):
        """
        Extracts notes from compressed files. Only used once. 
        """
        if (file_path.endswith('.tar.gz') or file_path.endswith('.tgz')):
            try:
                tar = tarfile.open(file_path)
                tar.extractall()
                tar.close()
            except:
                print("Error in extracting tar file")
        elif (file_path.endswith('.zip')):
            try:
                with ZipFile(file_path,'r') as zipObj:
                    zipObj.extractall()
            except:
                print("Error in extracting zip file")

    def create_vocab_dict(self):
        """
        Creates word2idx dictionary: word -> index
        Create idx2word dictionary: index -> word
        """
        self.word2idx = {word: index + 2 for index,word in enumerate(self.words_seen)}
        self.word2idx["UNK"] = 1 # Unknown words
        self.word2idx["PAD"] = 0 # Padding

        self.idx2word = {index: word for word,index in self.word2idx.items()}

        self.vocab_size = len(self.word2idx.keys()) 

    def create_label_dict(self):
        """
        Creates tag2idx dictionary: BIO label -> index
        Create idx2tag dictionary: index -> BIO label
        """
        self.tag2idx = {tag: index + 2 for index,tag in enumerate(self.tags_seen)}
        self.tag2idx["O"] = 1
        self.tag2idx["PAD"] = 0

        self.idx2tag = {index: tag for tag,index in self.tag2idx.items()}

        self.tag_size = len(self.tag2idx.keys())

    def process_and_get_data(self,train_folders):
        _, t_array, labels = self.process_data(train_folders)
        self.create_vocab_dict()
        self.create_label_dict()
        df = self.convert_to_df(t_array,labels) 
        X, y = self.create_train_set(df)
        self.save_processed_data(df)
        return X, y

    def process_text(self,root):
        """
        Processes the actual note. Tokenizes, adds to vocab, returns (1xsentences),(1xsentencesxtokens)
        """
        text_element = root.find('TEXT') # finds element with TEXT tag (i.e. the note)
        note: str = text_element.text
        note_sentences = sent_tokenize(note) # sentences
        note_tokens = list(map(lambda x: word_tokenize(x),note_sentences)) # sentences x tokens
        max_len = max(len(sent) for sent in note_tokens)
        if max_len > self.max_len:
            self.max_len = max_len
        self.words_seen.update([token for sent in note_tokens for token in sent]) # add tokens to vocab set
        return note_sentences, note_tokens

    def process_tags(self,root,note_tokens):
        """
        Processes tags for a document. Uses BIO system presented in Deep Learning paper.
        """
        labels = []
        tag_element = root.find('TAGS')
        tag_queue = [] # (token, tag)
        for tag in tag_element:
            attributes = tag.attrib
            label_tokens = word_tokenize(attributes['text'])
            for i, token in enumerate(label_tokens):
                if i == 0:
                    literal_tag ='B-'+attributes['TYPE']
                else:
                    literal_tag ='I-'+attributes['TYPE']
                tag_queue.append((token,literal_tag))
                self.tags_seen.add(literal_tag)

        next_token, next_tag = tag_queue.pop(0) 
        for sentence in note_tokens:
            label_sentence = []
            for token in sentence:
                if next_token != token:
                    label_sentence.append('O')
                else:
                    label_sentence.append(next_tag)
                    if len(tag_queue) > 0:
                        next_token, next_tag = tag_queue.pop(0)  
            labels.append(label_sentence)

        return labels

    def process_data(self,data_sets, is_train_set: bool = True):
        """
        Creates sentence and token vectors for all the files in a folder.
        """
        print("Processing data...")
        s_array = [] # documents x sentences
        t_array = [] # documents x sentences x tokens
        labels = [] # documents x sentences x tokens
        for dir_name in data_sets:
            for filename in os.listdir(dir_name):
                self.files_seen.append(filename)
                tree = ET.parse(dir_name + filename) # must pass entire path
                root = tree.getroot()
                note_sentences, note_tokens = self.process_text(root)
                s_array.append(note_sentences)
                t_array.append(note_tokens)
                if is_train_set:
                    labels.append(self.process_tags(root,note_tokens))
    
        return s_array, t_array, labels

    def convert_to_df(self,t_array,labels):
        """
        Converts data to pandas df. df contains docid, unpadded sentences, and unpadded tags.
        """
        data = []
        for i in range(len(t_array)): # documents
            docid = self.files_seen.pop(0)[:-4] # strips .xml from doc id 
            for j in range(len(t_array[i])): # sentences
                tokenized_sentence = t_array[i][j]
                label_sentence = labels[i][j]
                id_tokens = list(map(lambda token: self.word2idx[token],tokenized_sentence))
                id_labels = list(map(lambda label: self.tag2idx[label],label_sentence))
                data.append({'docid':docid,'sentence_w':tokenized_sentence,
                'sentence_i':id_tokens,'label_w':label_sentence,'label_i':id_labels})
        df = pd.DataFrame(data)
        return df

    def unstring_df_series(self,ids):
        """
        Unstrings a df series. Needed when you load data
        """
        temp = []
        for sentence in ids:
            temp.append(eval(sentence))
        return temp

    def create_train_set(self,df):
        """
        Creates training set using df by padding sequences and returning X,y.
        """
        # pad id'd sentences and tags
        sentence_ids = df["sentence_i"].copy()
        if type(sentence_ids[0]) is str:
            sentence_ids = self.unstring_df_series(sentence_ids)
        X = pad_sequences(maxlen=None, sequences=sentence_ids, dtype = 'int32', padding="post", value=self.word2idx["PAD"])

        label_ids = df["label_i"].copy()
        if type(label_ids[0]) is str:
            label_ids = self.unstring_df_series(label_ids)
        y = pad_sequences(maxlen=None, sequences=label_ids, padding="post", value=self.tag2idx["PAD"])

        print("Shape of X: ", X.shape)
        print("Shape of y: ", y.shape)

        return X, y

    def save_processed_data(self,df):
        folder = "small_data/"
        title = "small_data"
        path = folder + title
        if not os.path.exists(folder):
            os.makedirs(folder)
        if os.path.exists(path+'.xlsx'):
            os.remove(path+'.xlsx')
        df.to_excel(path+'.xlsx', sheet_name='PHI '+title)
        df.to_csv(path+'.csv')
        with open(path+'_word2idx.json','w') as f:
            json.dump(self.word2idx,f)
        with open(path+'_tag2idx.json','w') as f:
            json.dump(self.tag2idx,f)
        with open(path+'_idx2word.json','w') as f:
            json.dump(self.idx2word,f)
        with open(path+'_idx2tag.json','w') as f:
            json.dump(self.idx2tag,f)

    def load_processed_data(self,dir_name):
        df = None
        for filename in os.listdir(dir_name):
            path = dir_name + filename
            if filename.endswith('.csv'):
                df = pd.read_csv(path)
            if filename.endswith('word2idx.json'):
                with open(path) as f:
                    self.word2idx = json.load(f)
            if filename.endswith('tag2idx.json'):
                with open(path) as f:
                    self.tag2idx = json.load(f)
            if filename.endswith('idx2word.json'):
                with open(path) as f:
                    self.idx2word = json.load(f)
            if filename.endswith('idx2tag.json'):
                with open(path) as f:
                    self.idx2tag = json.load(f)
        return df

    def load_training_set(self,dir_name):
        df = self.load_processed_data(dir_name)
        return self.create_train_set(df)

        



