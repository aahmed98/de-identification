import tarfile
from zipfile import ZipFile
import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize, RegexpTokenizer
from keras.preprocessing.sequence import pad_sequences
import re
import json
import ast
from progressbar import ProgressBar
import time

PAD_IDX = 0
UNK_IDX = NON_PHI_IDX = 1

def extract_file(file_path:str):
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

def unstring_ids(dictionary):
    """
    Converts keys of dictionary to integers. Necessary because they are stored as string in JSON.
    """
    return {int(k):v for k,v in dictionary.items()}

def unstring_df_series(df_column: pd.Series):
    """
    Unstrings a df series. Needed when you load data.
    """
    pbar = ProgressBar()
    unstrung = []
    for sentence in pbar(df_column):
        unstrung.append(ast.literal_eval(sentence))
    return unstrung

def df_to_train_set(df: pd.DataFrame):
    """
    Creates training set using df by padding sequences and returning X,y. If loading, sentences should already be padded.
    """
    pbar = ProgressBar()
    
    # pad id'd sentences and tags
    sentence_ids = []
    sentence_groups = df.groupby(['docid','sentence'])['token_id']
    for _,data in pbar(sentence_groups):
        sentence_ids.append(data.to_numpy())
    X = pad_sequences(maxlen=None, sequences=sentence_ids, dtype = 'int32', padding="post", value=PAD_IDX)

    label_ids = []
    label_groups = df.groupby(['docid','sentence'])['label_id']
    for _,data in pbar(label_groups):
        label_ids.append(data.to_numpy())
    y = pad_sequences(maxlen=None, sequences=label_ids, dtype = 'int32', padding="post", value=PAD_IDX)

    print("Shape of X: ", X.shape)
    print("Shape of y: ", y.shape)

    return X, y

class PreProcessor:
    def __init__(self,title):
        self.words_seen = set()
        self.tags_seen = set()
        self.files_seen = [] #keeps track of doc id's
        
        self.max_len = 0
        self.title = title #title of folder to save data to

        self.tokenizer = RegexpTokenizer(r"[a-zA-Z0-9]+|[^a-zA-Z0-9\s]+") # modified tokenizer to exclude underscores as word characters

        self.tag_errors = []
    
    def replace_unknowns(self,tokens):
        """
        Replaces unseen words in test set with 'UNK' token
        """
        sentences = []
        for i in range(len(tokens)):
            sentence = []
            for j in range(len(tokens[i])):
                current_token = tokens[i][j]
                if current_token not in self.words_seen:
                    sentence.append("UNK")
                else:
                    sentence.append(current_token)
            sentences.append(sentence)
        return sentences

    def create_vocab_dict(self):
        """
        Creates word2idx dictionary: word -> index
        Create idx2word dictionary: index -> word
        """
        self.word2idx = {word: index + 2 for index,word in enumerate(self.words_seen)}
        self.word2idx["PAD"] = PAD_IDX # Padding
        self.word2idx["UNK"] = UNK_IDX # Unknown words

        self.idx2word = {index: word for word,index in self.word2idx.items()}

        self.vocab_size = len(self.word2idx.keys()) 

    def create_label_dict(self):
        """
        Creates tag2idx dictionary: BIO label -> index
        Create idx2tag dictionary: index -> BIO label
        """
        self.tag2idx = {tag: index + 2 for index,tag in enumerate(self.tags_seen)}
        self.tag2idx["PAD"] = PAD_IDX # PAD
        self.tag2idx["O"] = NON_PHI_IDX # non-PHI
        self.idx2tag = {index: tag for tag,index in self.tag2idx.items()}
        self.tag_size = len(self.tag2idx.keys())

    def my_tokenize(self,text):
        """
        Custom tokenization function. Handles several edge cases observed in the training data.
        """
        nltk_tokens = self.tokenizer.tokenize(text)
        new_tokens = []
        for token in nltk_tokens:
            capital_split = re.split(r"([a-z][A-Z][a-zA-Z]+|[a-zA-Z][A-Z][a-z])",token) # words connected without space (stageFour)
            num_chars = re.findall(r"\d+[a-zA-Z]+",token) # numbers connected to words (e.g. 78yom)
            char_num = re.findall(r"[a-zA-Z]+\d+",token)
            if len(capital_split) > 1: # two words are connected without a space
                new_tokens.append(capital_split[0] + capital_split[1][0]) # first word
                second_word = capital_split[1][1:]+capital_split[2]
                second_word_capital_split = re.split(r"([a-z][A-Z][a-zA-Z]+|[a-zA-Z][A-Z][a-z])",second_word)
                while len(second_word_capital_split) > 1: # recurse over second word
                    new_tokens.append(second_word_capital_split[0] + second_word_capital_split[1][0])
                    second_word = second_word_capital_split[1][1:]+second_word_capital_split[2]
                    second_word_capital_split = re.split(r"([a-z][A-Z][a-zA-Z]+|[a-zA-Z][A-Z][a-z])",second_word)
                new_tokens.append(second_word)
            elif len(num_chars) > 0: # number followed by letters
                num_char_split = re.findall(r"\d+|\D+",num_chars[0]) # splits numbers from letters
                for word in num_char_split:
                    new_tokens.append(word)
            elif len(char_num) > 0:
                char_num_split = re.findall(r"\d+|\D+",char_num[0]) # splits letters 
                for word in char_num_split:
                    new_tokens.append(word)
            else:
                new_tokens.append(token)
        return new_tokens

    def process_characters(self,note,tokens):
        """
        Gets the character range of each token.
        Params:
        note: raw string 
        tokens: sentences x tokens
        Returns:
        characters: sentences x tokens
        """
        current_token = ""
        characters = [] # list of tuples (start,end) of each token
        pointer = 0
        quotes = False
        for i in range(len(tokens)): #sentences
            sentence_chars = []
            for j in range(len(tokens[i])): #tokens
                current_token = tokens[i][j]
                if current_token in {"``","''"}: # weird edge cases with quotations
                    current_token = '"'
                    quotes = True
                window_size = len(current_token)
                while True:
                    note_window = note[pointer:pointer+window_size]
                    if note_window == current_token:
                        sentence_chars.append((pointer,pointer+window_size))
                        pointer += window_size
                        break
                    elif quotes and note[pointer:pointer+window_size+1] in {"``","''"}:
                        window_size += 1
                        sentence_chars.append((pointer,pointer+window_size))
                        pointer += window_size
                        break
                    pointer += 1 # new sentence
            characters.append(sentence_chars)
        return characters

    def process_text(self,note, isTrainSet = True):
        """
        Processes the actual note. Tokenizes, adds to vocab, returns (1xsentences),(1xsentencesxtokens)
        """
        #TODO: some text has dashes that are not tokenized. but the labels don't have those dashes.
        note_sentences = sent_tokenize(note) # sentences
        note_tokens = list(map(lambda x: self.my_tokenize(x),note_sentences)) # sentences x tokens
        note_characters = self.process_characters(note,note_tokens) # sentences x tokens
        max_len = max(len(sent) for sent in note_tokens)
        if max_len > self.max_len:
            self.max_len = max_len
        if isTrainSet:
            self.words_seen.update([token for sent in note_tokens for token in sent]) # add tokens to vocab set

        return note_sentences, note_tokens, note_characters

    def process_tags(self,root,note_tokens,filename = None):
        """
        Processes tags for a document. Uses BIO system common in NER.
        """
        tags = root.find('TAGS')
        tag_queue = [] # (token, tag)
        for tag in tags:
            attributes = tag.attrib
            label_tokens = self.my_tokenize(attributes['text'])
            for i, token in enumerate(label_tokens): # seperate tags into tokens for B,I purposes
                if i == 0:
                    literal_tag ='B-'+attributes['TYPE']
                else:
                    literal_tag ='I-'+attributes['TYPE']
                tag_queue.append((token,literal_tag))
                self.tags_seen.add(literal_tag)

        num_tags_actual = len(tag_queue)
        num_tags_counted = 0

        labels = []
        next_tag_token, next_tag = tag_queue.pop(0) # next_token is next token to have PHI
        consecutive = False # checks for errors in preprocessing
        old_tag, old_tag_token = None, None
        for sentence in note_tokens:
            label_sentence = []
            for token in sentence:
                if next_tag_token != token:
                    if consecutive: # wrong label
                        if len(label_sentence) > 0: # wrong label is in current sentence
                            label_sentence.pop()
                            label_sentence.append('O')
                        else: # wrong label was in previous sentence
                            labels[-1].pop()
                            labels[-1].append('O')
                        num_tags_counted -= 1
                        tag_queue.insert(0,(next_tag_token,next_tag)) # insert I-___ back into queue
                        next_tag, next_tag_token = old_tag, old_tag_token # reset next_tag to B-___
                        consecutive = False
                    label_sentence.append('O')
                else:
                    label_sentence.append(next_tag)
                    num_tags_counted += 1
                    if len(tag_queue) > 0:
                        old_tag, old_tag_token = next_tag, next_tag_token
                        next_tag_token, next_tag = tag_queue.pop(0)
                        if old_tag[1:] == next_tag[1:] and old_tag[0] == "B" and next_tag[0] == "I": # consecutive tags
                            consecutive = True
                        else:
                            consecutive = False
                    else:
                        consecutive = False

            labels.append(label_sentence)

        matched = num_tags_actual == num_tags_counted
        # asserts that all tags in tag queue were processed correctly

        return labels, matched 

    def process_data(self,data_sets, isTrainSet: bool = True):
        """
        Creates sentence and token vectors for all the files in a folder.
        """
        print("Preprocessing data...")
        pbar = ProgressBar()

        s_array = [] # documents x sentences
        t_array = [] # documents x sentences x tokens
        c_array = [] # documents x sentences x tokens
        labels = [] # documents x sentences x tokens
        for dir_name in pbar(data_sets):
            for filename in os.listdir(dir_name):
                # print(filename)
                tree = ET.parse(dir_name + filename) # must pass entire path
                root = tree.getroot()
                note = root.find("TEXT").text
                note_sentences, note_tokens, note_characters = self.process_text(note, isTrainSet)
                tags, matched = self.process_tags(root,note_tokens, filename) 
                if not isTrainSet:
                    note_tokens = self.replace_unknowns(note_tokens)
                if not matched:
                    self.tag_errors.append(filename)
                    continue
                self.files_seen.append(filename)   
                s_array.append(note_sentences)
                t_array.append(note_tokens)
                c_array.append(note_characters)
                labels.append(tags)

        print("# of Tag Processing Errors: ",len(self.tag_errors))
        print("Files with errors: ",self.tag_errors)
        return s_array, t_array,c_array, labels

    def create_df(self,t_array,c_array,labels):
        """
        Converts data to pandas df. df contains docid, unpadded sentences, and unpadded tags.
        """
        data = []
        for i in range(len(t_array)): # documents
            docid = self.files_seen.pop(0)[:-4] # strips .xml from doc id 
            for j in range(len(t_array[i])): # sentences
                tokenized_sentence = t_array[i][j]
                character_sentence = c_array[i][j]
                label_sentence = labels[i][j]
                assert len(tokenized_sentence) == len(character_sentence) == len(label_sentence), "counted characters or tokens incorrectly"
                for k in range(len(tokenized_sentence)): # loop through tokens
                    token = tokenized_sentence[k]
                    token_id = self.word2idx[token]
                    label = label_sentence[k]
                    label_id = self.tag2idx[label]
                    characters = character_sentence[k]
                    data.append({'docid':docid,'sentence':j, 'token': token,
                    'token_id':token_id,'label':label,'label_id':label_id,
                    'characters':characters})

        df = pd.DataFrame(data)
        return df

    def unstring_ids(self,dictionary):
        """
        Converts keys of dictionary to integers. Necessary because they are stored as string in JSON.
        """
        return {int(k):v for k,v in dictionary.items()}

    def unstring_df_series(self,ids):
        """
        Unstrings a df series. Needed when you load data.
        """
        temp = []
        for sentence in ids:
            temp.append(eval(sentence))
        return temp

    def create_train_set(self,df):
        """
        Creates training set using df by padding sequences and returning X,y. If loading, sentences are already padded.
        """
        X,y = df_to_train_set(df)
        self.max_len = y.shape[1]
        return X, y

    def save_processed_data(self,df):
        """
        Saves df to csv/excel and dictionaries to json
        """
        title = self.title
        folder = "../data/preprocessed/" + title + "/"
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
        """
        Directory contains csv file and dictionaries as json
        """
        print("Loading preprocessed data...")
        df = None
        if not dir_name.endswith('/'):
            dir_name = dir_name + "/"
        for filename in os.listdir(dir_name):
            path = dir_name + filename
            if filename.endswith('.csv'):
                df = pd.read_csv(path)
            if filename.endswith('word2idx.json'):
                with open(path) as f:
                    self.word2idx = json.load(f)
                    self.vocab_size = len(self.word2idx.keys()) 
            if filename.endswith('tag2idx.json'):
                with open(path) as f:
                    self.tag2idx = json.load(f)
                    self.tag_size = len(self.tag2idx.keys())
            if filename.endswith('idx2word.json'):
                with open(path) as f:
                    self.idx2word = self.unstring_ids(json.load(f))
            if filename.endswith('idx2tag.json'):
                with open(path) as f:
                    self.idx2tag = self.unstring_ids(json.load(f))
        return df

    def load_test_data(self,test_dir_name):
        """
        Directory contains csv file and dictionaries as json.
        """
        print("Loading preprocessed test data...")
        df = None
        if not test_dir_name.endswith('/'):
            test_dir_name = test_dir_name + "/"
        for filename in os.listdir(test_dir_name):
            path = test_dir_name + filename
            if filename.endswith('.csv'):
                df = pd.read_csv(path)
        return df

    def save_test_data(self, title, df):
        """
        Saves df to csv/excel and dictionaries to json
        """
        folder = "../data/preprocessed/" + title + "/"
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


    def get_data(self,train_folders,isLoading = False):
        """
        All-purpose function to get data.
        isLoading: train_folder is SINGLE path to dir that contains .csv, dictionaries.
        !isLoading: rain_folders is LIST of paths to dirs that contain i2b2 data 
        """
        if not isLoading:
            _, t_array,c_array,labels = self.process_data(train_folders, isTrainSet=True)
            self.create_vocab_dict()
            self.create_label_dict()
            df = self.create_df(t_array,c_array,labels)
            X, y = self.create_train_set(df) # modifies df, which is why it comes before sav
            self.save_processed_data(df)
        else:
            df = self.load_processed_data(train_folders)
            X,y = self.create_train_set(df) # no modification to df in loading case
        print("Preprocessing complete.")
        return X, y, df

    def create_test_set(self,test_folders, isLoading = False, title = None):
        """
        Creates test set given test folders.
        """
        if not isLoading:
            _, t_array,c_array,labels = self.process_data(test_folders,isTrainSet=False)
            df = self.create_df(t_array,c_array,labels)
            X, y = df_to_train_set(df)
            self.save_test_data(title,df)
        else:
            df = self.load_test_data(test_folders)
            X, y = df_to_train_set(df)
        return X,y,df

if __name__ == "__main__":
    train_folders = ["../../data/raw/training-PHI-Gold-Set2/"]
    # train_folders = ["../../data/testing/"]
    pp = PreProcessor("testing_full")
    _, t_array,c_array,labels = pp.process_data(train_folders, isTrainSet=True)
    pp.create_vocab_dict()
    pp.create_label_dict()
    df = pp.create_df(t_array,c_array,labels)
    print(df.head())
    X, y = pp.create_train_set(df)
    pp.save_processed_data(df)
    print(X)
    print(y)