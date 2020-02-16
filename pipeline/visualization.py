from random import randint
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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

def loss_plot(losses,title):
    """
    Generates and saves a plot of training loss
    """
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.title(title)
    plt.savefig(title+'.png')
    plt.close()