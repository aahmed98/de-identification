import tarfile
from zipfile import ZipFile
import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

class PreProcessor:
    def __init__(self):
        self.vocab_set = set()
        self.vocab_dict = {}
        self.files_seen = []

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
        Creates vocab dictionary: word -> index
        """
        self.vocab_dict = {word: index for index,word in enumerate(self.vocab_set)}

    def get_data(self,train_folders, test_unlabeled, test_labeled):
        pass

    def process_text(self,root):
        """
        Processes the actual note. Tokenizes, adds to vocab, returns (1xsentences),(1xsentencesxtokens)
        """
        text_element = root.find('TEXT') # finds element with TEXT tag (i.e. the note)
        note: str = text_element.text
        note_sentences = sent_tokenize(note) # sentences
        note_tokens = list(map(lambda x: word_tokenize(x),note_sentences)) # sentences x tokens
        self.vocab_set.update([token for sent in note_tokens for token in sent]) # add tokens to vocab set
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
                    tag_queue.append((token,'B-'+attributes['TYPE']))
                else:
                    tag_queue.append((token,'I-'+attributes['TYPE']))

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

    def process_data(self,dir_name: str, is_train_set: bool = True):
        """
        Creates sentence and token vectors for all the files in a folder.
        """
        print("Processing data...")
        i = 0
        s_array = [] # documents x sentences
        t_array = [] # documents x sentences x tokens
        labels = [] # documents x sentences x tokens
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

    def create_train_set(self,t_array,labels):
        """
        Converts data to pandas df. 
        """
        data = []
        for i in range(len(t_array)): # documents
            docid = self.files_seen.pop(0)[:-4] # strips .xml from doc id 
            for j in range(len(t_array[i])): # sentences
                tokenized_sentence = t_array[i][j]
                label_sentence = labels[i][j]
                id_sentence = list(map(lambda token: self.vocab_dict[token],tokenized_sentence))
                data.append({'docid':docid,'sentence':id_sentence,'label':label_sentence})
        df = pd.DataFrame(data)
        print(df.head())
        print(df.info())
        print(df.describe())
        print(df['label'].value_counts())



