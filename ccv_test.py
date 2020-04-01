# import preprocessing code
from src.preprocess import PreProcessor, df_to_train_set

# save paths to the available datasets
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
    preprocessed_folder = "../de-ID_data/preprocessed/sample_data/",
    raw_folders = ["docs/Track1-de-indentification/PHI/"]
)

GOLD_1 = Dataset(
    title = "gold_1",
    preprocessed_folder = "../de-ID_data/preprocessed/gold_1/",
    raw_folders = ["../de-ID_data/raw/training-PHI-Gold-Set1/"]
)

GOLD_FULL = Dataset(
    title = "gold_full",
    preprocessed_folder = "../de-ID_data/preprocessed/gold_full/",
    raw_folders = ["../de-ID_data/raw/training-PHI-Gold-Set1/","../data/raw/training-PHI-Gold-Set2/"]
)

GOLD_TEST = Dataset(
    title = "gold_test",
    preprocessed_folder = "../de-ID_data/preprocessed/gold_test/",
    raw_folders = ["../de-ID_data/raw/testing-PHI-Gold-fixed/"]
)

DATASETS = [SAMPLE_DATA,GOLD_1,GOLD_FULL, GOLD_TEST]

# pick dataset and define loading boolean
train_data = DATASETS[2]
# train_data = DATASETS[0]
test_data = DATASETS[3]
isLoading = True

# save paths to the available datasets
from typing import NamedTuple, List

# attach data to PreProcessor object.
pp = PreProcessor(train_data.title)
if isLoading:
    X_train,y_train,df_train = pp.get_data(train_data.preprocessed_folder,isLoading = isLoading)
else:
    X_train,y_train,df_train = pp.get_data(train_data.raw_folders,isLoading = isLoading)
print("max length: ",pp.max_len)

# load test set
if isLoading:
    X_test,y_test,df_test = pp.create_test_set(test_data.preprocessed_folder,isLoading,test_data.title)
else:
    X_test,y_test,df_test = pp.create_test_set(test_data.raw_folders,isLoading,test_data.title)

# import model stuff
from src.models.baseline import BaselineModel
from src.models.bilstm import BiLSTM
from src.models.bilstm_crf import BiLSTM_CRF
from src.models.transformer import Transformer
from src.models.transformer_crf import Transformer_CRF
from src.models.bilstm_chars import BiLSTM_Chars
from pipeline.visualization import sample_output
from pipeline.train import train
from random import randint
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from src.converter import get_label_positions, bio_to_i2d2
import xml.etree.ElementTree as ET
from typing import NamedTuple, List

# check if GPU is available
assert tf.test.is_built_with_cuda()
physical_devices = tf.config.list_physical_devices('GPU') 
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num GPUs:", len(physical_devices)) 

tf.config.gpu.set_per_process_memory_growth(True)
tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)

# build model
# model = BaselineModel(pp.vocab_size,pp.tag_size,pp.max_len)
# model = BiLSTM(pp.vocab_size,pp.tag_size,pp.max_len)
# model = BiLSTM_CRF(pp.vocab_size,pp.tag_size,pp.max_len)
# model = Transformer(pp.vocab_size,pp.tag_size,pp.max_len)
# model = Transformer_CRF(pp.vocab_size, pp.tag_size, pp.max_len)
model = BiLSTM_Chars(pp.vocab_size, pp.tag_size, pp.max_len,pp.idx2word)

# configure checkpoints and checkpoint manager
checkpoint_dir = 'models/checkpoints/' + train_data.title + '/' + model.title + '/' 
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=10)

# restore checkpoint
checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))

# train
print("Training ",model.title)
losses = train(model,X_train,y_train,batch_size = 16, epochs=10, lr = 0.0005, sample_interval=10,manager=manager,pp=pp)

# sample a random output
sample_output(model,X_train,y_train, pp = pp,rand_idx=None)

# test model
from pipeline.test import test_to_i2d2

# test_to_i2d2(model,df_test, pp, checkpoint, manager)

