from preprocess import PreProcessor
from baseline import BaselineModel
from random import randint
from sklearn.utils import shuffle
import tensorflow as tf
import os
import matplotlib.pyplot as plt

DATASETS = [
    ("sample_data",["../Track1-de-indentification/PHI/"]),
    ("gold_1",["../training-PHI-Gold-Set1/"]),
    ("gold_full",["../training-PHI-Gold-Set1/","../training-PHI-Gold-Set2/"])
]

def sample_output(pp: PreProcessor, model, train_inputs, train_labels):
    n = len(train_inputs)
    rand_idx = randint(0,n)
    print("Sentence #: ",rand_idx)
    sample_input = train_inputs[rand_idx]
    sample_labels = train_labels[rand_idx]
    sample_input_reshaped = tf.reshape(sample_input,(1,-1))
    predicted_output = model(sample_input_reshaped)
    mle_output = tf.argmax(predicted_output,axis=2).numpy().flatten()

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

def train(model, manager, train_inputs, train_labels, batch_size = 32,epochs= 10):
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
                probs = model(input_batch)
                loss = model.loss(probs,output_batch)

            epoch_loss += loss
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            if i % (batch_size*5) == 0:
                print("Loss after batch ",int(i/batch_size), ": ",loss)
        print("LOSS: ",epoch_loss)
        losses.append(epoch_loss)
        manager.save() # save checkpoint at end of epoch
    
    title = model.title
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.title(title)
    plt.savefig(title+'.png')
    plt.close()

    #sample_output(pp,model,train_inputs,train_labels)

def test(pp,model,checkpoint,manager,test_inputs,test_labels):
    print("Loading checkpoint...")
    #checkpoint.restore(manager.latest_checkpoint)
    for _ in range(10):
        sample_output(pp,model,test_inputs,test_labels)

def main():
    sample_data = DATASETS[0]
    pp = PreProcessor(sample_data[0])
    train_folder = sample_data[1]
    load_folder = sample_data[0]
    X,y = pp.get_data(load_folder,True)

    # model = BaselineModel(pp.vocab_size,pp.tag_size,pp.max_len)
    # checkpoint_dir = './checkpoints'
    # checkpoint = tf.train.Checkpoint(model=model)
    # manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    # train(model, manager, X,y,epochs=50)
    # test(pp,model,checkpoint,manager,X,y)

if __name__ == "__main__":
    main()