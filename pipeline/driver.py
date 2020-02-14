from preprocess import PreProcessor, df_to_train_set
from baseline import BaselineModel
from bilstm import BiLISTM
from bilstm_crf import BiLISTM_CRF
from random import randint
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from converter import get_label_positions, bio_to_i2d2
import xml.etree.ElementTree as ET

DATASETS = [
    ("sample_data",["../Track1-de-indentification/PHI/"]),
    ("gold_1",["../training-PHI-Gold-Set1/"]),
    ("gold_full",["../training-PHI-Gold-Set1/","../training-PHI-Gold-Set2/"])
]

def predict_document(model,docid,df):
    """
    df contains all sentences of docid.
    """
    unique_docids = df["docid"].unique()
    assert docid in unique_docids, "DocID not in DataFrame"
    doc_sentences = df.groupby(by="docid").get_group(docid) # dataframe
    X,_ = df_to_train_set(doc_sentences,True)
    predictions = tf.argmax(model(X),axis=2).numpy()
    return predictions

def sample_output(model, train_inputs, train_labels, pp, df=None,rand_idx = None):
    if rand_idx is None:
        n = len(train_inputs)
        rand_idx = randint(0,n)
    print("Sentence #: ",rand_idx)
    sample_input = train_inputs[rand_idx]
    sample_labels = train_labels[rand_idx]
    if df is not None:
        print(df.iloc[rand_idx,:6])
    sample_input_reshaped = tf.reshape(sample_input,(1,-1))
    predicted_output = model(sample_input_reshaped)
    mle_output = tf.argmax(predicted_output,axis=2).numpy().flatten()

    orig_sentence = [pp.idx2word[idx] for idx in sample_input]
    true_tags = [pp.idx2tag[idx] for idx in sample_labels]
    predicted_tags = [pp.idx2tag[idx] for idx in mle_output]

    fancy_print(orig_sentence,predicted_tags,true_tags)

def sample_output_CRF(model, train_inputs, train_labels, pp, df=None,rand_idx = None):
    if rand_idx is None:
        n = len(train_inputs)
        rand_idx = randint(0,n)
    print("Sentence #: ",rand_idx)
    sample_input = train_inputs[rand_idx]
    sample_labels = train_labels[rand_idx]
    if df is not None:
        print(df.iloc[rand_idx,:6])
    sample_input_reshaped = tf.reshape(sample_input,(1,-1))
    predicted_output = model.predict(sample_input_reshaped)
    mle_output = np.array(predicted_output).flatten()
    print(mle_output)

    orig_sentence = [pp.idx2word[idx] for idx in sample_input]
    true_tags = [pp.idx2tag[idx] for idx in sample_labels]
    predicted_tags = [pp.idx2tag[idx] for idx in mle_output]

    fancy_print(orig_sentence,predicted_tags,true_tags)

def fancy_print(input_sentence, predictions, labels):
    """
    Prints sentence word by word with predicted and true tags alongside them.
    """
    print("{:15} {:5}: ({})".format("Word", "Pred", "True"))
    print("="*30)
    for w, true, pred in zip(input_sentence, predictions, labels):
        if w != "PAD":
            print("{:15}:{:5} ({})".format(w, true, pred))

def train(model, train_inputs, train_labels, batch_size = 32,epochs= 10, sample_interval = 5, pp = None, manager = None):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    n = len(train_inputs)
    losses = []
    for epoch in range(epochs):
        print("--------- EPOCH ",epoch,"-----------")
        X, y = shuffle(train_inputs,train_labels)
        epoch_loss = 0
        for i in range(0,n,batch_size):
            input_batch = X[i:i+batch_size]
            output_batch = y[i:i+batch_size]
            with tf.GradientTape() as tape:
                probs = model(input_batch) # bsz x max_len x tag_size
                loss = model.loss(probs,output_batch)

            epoch_loss += loss
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            if i % (batch_size*5) == 0:
                print("Epoch: %d, Batch: %d, Loss: %f" % (epoch,int(i/batch_size),loss))
        losses.append(epoch_loss)
        if manager is not None:
            manager.save() # save checkpoint at end of epoch
        if pp is not None and epoch % sample_interval == 0:
            sample_output(model,train_inputs,train_labels,pp)
    title = model.title
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.title(title)
    plt.savefig(title+'.png')
    plt.close()
    sample_output(pp,model,train_inputs,train_labels)


def train_CRF(model, train_inputs, train_labels, batch_size = 32,epochs= 10, sample_interval = 5, pp = None, manager = None):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    n = len(train_inputs)
    losses = []
    for epoch in range(epochs):
        print("--------- EPOCH ",epoch,"-----------")
        X, y = shuffle(train_inputs,train_labels)
        epoch_loss = 0
        for i in range(0,n,batch_size):
            input_batch = X[i:i+batch_size]
            output_batch = y[i:i+batch_size]
            with tf.GradientTape() as tape:
                log_likelihood,transition = model(input_batch,output_batch) # bsz x max_len x tag_size
                loss = model.loss(log_likelihood)

            epoch_loss += loss
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            if i % (batch_size*5) == 0:
                print("Epoch: %d, Batch: %d, Loss: %f" % (epoch,int(i/batch_size),loss))
        losses.append(epoch_loss)
        if manager is not None:
            manager.save() # save checkpoint at end of epoch
        if pp is not None and epoch % sample_interval == 0:
            sample_output_CRF(model,train_inputs,train_labels,pp)
    title = model.title
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.title(title)
    plt.savefig(title+'.png')
    plt.close()

    sample_output_CRF(pp,model,train_inputs,train_labels)

def test(model,test_inputs,test_labels,pp,df,checkpoint = None,manager= None):
    if checkpoint is not None and manager is not None:
        print("Loading checkpoint...")
        checkpoint.restore(manager.latest_checkpoint)

    # tree = ET.parse(DATASETS[0][1][0] + "320-01.xml") # must pass entire path
    # root = tree.getroot()
    # note = root.find('TEXT').text
    # predictions = predict_document(model,'320-01',df)
    # doc_labels = get_label_positions(df,'320-01',predictions)
    # print(doc_labels)
    # bio_to_i2d2(df,doc_labels,note)

    # for _ in range(10):
    #     sample_output(model,test_inputs,test_labels,pp,df)
    for i in [43,105]:
        sample_output(model,test_inputs,test_labels,pp,df,i)

def main():
    # LOAD DATA
    sample_data = DATASETS[0]
    pp = PreProcessor(sample_data[0]) # PreProcesser attached to data. Contains dictionaries, max_len, vocab_size, etc.
    load_folder = sample_data[0]
    X,y,df = pp.get_data(load_folder,True)
    # train_folder = sample_data[1]
    # X,y,df = pp.get_data(train_folder,False)
    print("max length: ",pp.max_len)
    
    # CREATE MODEL AND CHECKPOINTS
    # model = BaselineModel(pp.vocab_size,pp.tag_size,pp.max_len)
    # checkpoint_dir = load_folder + '/checkpoints'
    model = BiLISTM_CRF(pp.vocab_size,pp.tag_size,pp.max_len)
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)
    # checkpoint = tf.train.Checkpoint(model=model)
    # manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    # TRAIN AND TEST
    # train(model,X,y,epochs=100,sample_interval=10,manager=manager,pp=pp)
    train_CRF(model,X,y,epochs=100,sample_interval=10,pp=pp)
    # test(model,X,y,pp,df,checkpoint,manager)

if __name__ == "__main__":
    main()