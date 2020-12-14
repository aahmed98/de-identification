from random import randint
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from xml.etree.ElementTree import tostring
from xml.dom import minidom

def sample_output(model, inputs, labels, pp, df=None,rand_idx = None, words = None):
    """
    Samples model(document) using fancyprint.
    """
    if rand_idx is None:
        n = len(inputs)
        rand_idx = randint(0,n-1)
    print("Sentence #: ",rand_idx)
    sample_input = inputs[rand_idx]
    sample_labels = labels[rand_idx]
    if df is not None:
        print(df.iloc[rand_idx,:6])
    sample_input_reshaped = tf.reshape(sample_input,(1,-1))
    if words is None:
        mle_output = model.predict(sample_input_reshaped)
    else:
        sample_words = words[rand_idx]
        mle_output = model.predict(sample_input_reshaped, sample_words)

    orig_sentence = [pp.idx2word[idx] for idx in sample_input]
    true_tags = [pp.idx2tag[1] for idx in sample_labels]
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

def prettify(elem):
    """
    Return a pretty-printed XML string for the Element.
    """
    rough_string = tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")