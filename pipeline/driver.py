from src.preprocess import PreProcessor, df_to_train_set
from src.models.baseline import BaselineModel
from src.models.bilstm import BiLSTM
from src.models.bilstm_crf import BiLSTM_CRF
from .visualization import sample_output
from .train import train_CRF
from random import randint
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from src.converter import get_label_positions, bio_to_i2d2
import xml.etree.ElementTree as ET

from typing import NamedTuple, List


class Dataset(NamedTuple):
    """
    Interface for accessing data folders.
    """
    title: str
    preprocessed_folder: str
    raw_folders: List[str]

SAMPLE_DATA = Dataset(
    title = "sample_data",
    preprocessed_folder = "../data/preprocessed/sample_data/",
    raw_folders = ["docs/Track1-de-indentification/PHI/"]
)

GOLD_1 = Dataset(
    title = "gold_1",
    preprocessed_folder = "../data/preprocessed/gold_1/",
    raw_folders = ["../data/raw/training-PHI-Gold-Set1/"]
)

GOLD_FULL = Dataset(
    title = "gold_full",
    preprocessed_folder = "../data/preprocessed/gold_full/",
    raw_folders = ["../data/raw/training-PHI-Gold-Set1/","../data/raw/training-PHI-Gold-Set2/"]
)

DATASETS = [SAMPLE_DATA,GOLD_1,GOLD_FULL]

def predict_document(model,docid,df):
    """
    Given a model, predict the PHI for all the sentences in docid (docid must exist in df)
    df contains all sentences of docid.
    """
    unique_docids = df["docid"].unique()
    assert docid in unique_docids, "DocID not in DataFrame"
    doc_df = df.groupby(by="docid").get_group(docid) # dataframe
    print(doc_df)
    X,_ = df_to_train_set(doc_df,True)
    predictions = tf.argmax(model(X),axis=2).numpy()
    return predictions, doc_df

def test(model,test_inputs,test_labels,pp,df,checkpoint = None,manager= None):
    """
    Predicts the PHI for all the documents in a DataFrame. Writes them to evaluation_data in i2b2 format.
    """
    if checkpoint is not None and manager is not None:
        print("Loading checkpoint...")
        checkpoint.restore(manager.latest_checkpoint)

    unique_docids = df["docid"].unique()
    for docid in unique_docids:
        print("Doc ID: ",docid)
        tree = ET.parse(SAMPLE_DATA.raw_folders[0] + docid + ".xml") # must pass entire path
        root = tree.getroot()
        note = root.find('TEXT').text
        predictions, doc_df = predict_document(model,docid,df)
        doc_labels = get_label_positions(predictions,pp.idx2tag)
        # print(doc_labels)
        xml_doc = bio_to_i2d2(doc_df,doc_labels,note)
        ET.ElementTree(xml_doc).write("evaluation_data/" + model.title + "/"+ docid+".xml")

    for i in [43,105]:
        sample_output(model,test_inputs,test_labels,pp,df,i)

def main():
    """
    Driver code.
    """
    # LOAD DATA
    data = DATASETS[2]
    isLoading = True
    pp = PreProcessor(data.title) # PreProcesser attached to data. Contains dictionaries, max_len, vocab_size, etc.
    if isLoading:
        X,y,df = pp.get_data(data.preprocessed_folder,isLoading = True)
    else:
        X,y,df = pp.get_data(data.raw_folders,isLoading = False)
    print("max length: ",pp.max_len)
    
    # CREATE MODEL AND CHECKPOINTS
    # model = BiLSTM_CRF(pp.vocab_size,pp.tag_size,pp.max_len)
    # model = BaselineModel(pp.vocab_size,pp.tag_size,pp.max_len)
    # checkpoint_dir = 'models/checkpoints/' + model.title + '/' 
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)
    # checkpoint = tf.train.Checkpoint(model=model)
    # manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    # # TRAIN AND TEST
    # # train(model,X,y,epochs=100,sample_interval=10,manager=manager,pp=pp)
    # # train_CRF(model,X,y,epochs=100,sample_interval=10,pp=pp)
    # test(model,X,y,pp,df,checkpoint,manager)

if __name__ == "__main__":
    main()