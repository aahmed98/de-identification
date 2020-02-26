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
import wordninja
from progressbar import ProgressBar

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

def unstring_df_series(df_column):
    """
    Unstrings a df series. Needed when you load data.
    """
    temp = []
    for sentence in df_column:
        temp.append(eval(sentence))
    return temp

def df_to_train_set(df: pd.DataFrame, loading = False):
    """
    Creates training set using df by padding sequences and returning X,y. If loading, sentences should already be padded.
    """
    if not loading: # NOT loading
        # pad id'd sentences and tags
        sentence_ids = df["sentence_ids"].copy()
        if type(sentence_ids[0]) is str:
            sentence_ids = unstring_df_series(sentence_ids)
        X = pad_sequences(maxlen=None, sequences=sentence_ids, dtype = 'int32', padding="post", value=PAD_IDX)
        df["padded_sentence"] = X.tolist()

        label_ids = df["labels_ids"].copy()
        if type(label_ids[0]) is str:
            label_ids = unstring_df_series(label_ids)
        y = pad_sequences(maxlen=None, sequences=label_ids, padding="post", value=PAD_IDX)
        df["padded_labels"] = y.tolist()

    else: # loading
        X = df["padded_sentence"].copy()
        if type(X.iloc[0]) is str: # lists converted to strings when you save. must "unpack" string
            X = np.array(unstring_df_series(X))
        y = df["padded_labels"].copy()
        if type(y.iloc[0]) is str:
            y = np.array(unstring_df_series(y))

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
        Custom tokenization function. Handles several edge cases observed in training data.
        """
        return word_tokenize(text)
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
                    print("Recursing! Original word: ",token, "Second compound word: ",second_word)
                    new_tokens.append(second_word_capital_split[0] + second_word_capital_split[1][0])
                    second_word = second_word_capital_split[1][1:]+second_word_capital_split[2]
                    second_word_capital_split = re.split(r"([a-z][A-Z][a-zA-Z]+|[a-zA-Z][A-Z][a-z])",second_word)
                # new_tokens.append(capital_split[1][1:]+capital_split[2]) # second word
                new_tokens.append(second_word)
            elif len(num_chars) > 0: # number followed by letters
                num_char_split = re.findall(r"\d+|\D+",num_chars[0]) # splits numbers from letters
                for word in num_char_split:
                    new_tokens.append(word)
            elif len(char_num) > 0:
                char_num_split = re.findall(r"\d+|\D+",char_num[0]) # splits letters 
                for word in char_num_split:
                    new_tokens.append(word)
                # print(char_num_split)
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

    def process_text(self,note):
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
        # print(tag_queue)

        num_tags_counted = 0
        labels = []
        next_tag_token, next_tag = tag_queue.pop(0) # next_token is next token to have PHI
        printing = False
        if filename == "400-05.xml":
            printing = False
        consecutive = False # checks for errors in preprocessing
        old_tag, old_tag_token = None, None
        for sentence in note_tokens:
            # if printing:
            #     print(tag_queue, "\n")
            label_sentence = []
            for token in sentence:
                if printing:
                    print("next tag token: ",next_tag_token)
                    print("current token: ",token)
                if next_tag_token != token:
                    if consecutive: # wrong label
                        if printing:
                            print("Sentence: ",sentence)
                            print("Next Tag Token: ",next_tag_token)
                            print("Next Token: ",token)
                            print("Old Tag Token: ",old_tag_token)
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
                    if printing:
                        print("Appending",next_tag,"to the tag list")
                    label_sentence.append(next_tag)
                    num_tags_counted += 1
                    if len(tag_queue) > 0:
                        old_tag, old_tag_token = next_tag, next_tag_token
                        next_tag_token, next_tag = tag_queue.pop(0)
                        if printing:
                            print(next_tag_token,next_tag)
                        if old_tag[1:] == next_tag[1:] and old_tag[0] == "B" and next_tag[0] == "I": # consecutive tags
                            consecutive = True
                        else:
                            consecutive = False
                    else:
                        break

            labels.append(label_sentence)

        # print(labels)

        matched = num_tags_actual == num_tags_counted
        # asserts that all tags in tag queue were processed correctly
        # print("Matched? ",matched) # "Mismatch. Actual #: "+ str(num_tags_actual) + ", Counted #: " + str(num_tags_counted)
        if printing:
            print(labels)

        return labels, matched 

    def process_data(self,data_sets, is_train_set: bool = True):
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
                self.files_seen.append(filename)
                # print(filename)
                tree = ET.parse(dir_name + filename) # must pass entire path
                root = tree.getroot()
                note = root.find("TEXT").text
                note_sentences, note_tokens, note_characters = self.process_text(note)
                s_array.append(note_sentences)
                t_array.append(note_tokens)
                c_array.append(note_characters)
                if is_train_set:
                    tags, matched = self.process_tags(root,note_tokens, filename) 
                    labels.append(tags)
                    if not matched:
                        self.tag_errors.append(filename)

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
                id_tokens = list(map(lambda token: self.word2idx[token],tokenized_sentence))
                id_labels = list(map(lambda label: self.tag2idx[label],label_sentence))
                data.append({'docid':docid,'sentence':tokenized_sentence,
                'sentence_ids':id_tokens,'labels':label_sentence,'labels_ids':id_labels,
                'characters':character_sentence})
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

    def create_train_set(self,df,loading=False):
        """
        Creates training set using df by padding sequences and returning X,y. If loading, sentences are already padded.
        """
        X,y = df_to_train_set(df,loading)
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

    def get_data(self,train_folders,isLoading = False):
        """
        All-purpose function to get data.
        isLoading: train_folder is SINGLE path to dir that contains .csv, dictionaries.
        !isLoading: rain_folders is LIST of paths to dirs that contain i2b2 data 
        """
        if not isLoading:
            _, t_array,c_array,labels = self.process_data(train_folders)
            self.create_vocab_dict()
            self.create_label_dict()
            df = self.create_df(t_array,c_array,labels)
            X, y = self.create_train_set(df,isLoading) # modifies df, which is why it comes before sav
            self.save_processed_data(df)
        else:
            df = self.load_processed_data(train_folders)
            X,y = self.create_train_set(df,isLoading) # no modification to df in loading case
        print("Preprocessing complete.")
        return X, y, df