from preprocess import PreProcessor
from baseline import BaselineModel
from random import randint
from sklearn.utils import shuffle
import tensorflow as tf
import os
import matplotlib.pyplot as plt

def fancy_print(input_sentence, predictions, labels):
    print("{:15} {:5}: ({})".format("Word", "Pred", "True"))
    print("="*30)
    for w, true, pred in zip(input_sentence, predictions, labels):
        if w != "__PAD__":
            print("{:15}:{:5} ({})".format(w, true, pred))

def train(pp, model, manager, train_inputs, train_labels, batch_size = 32,epochs= 10):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    n = len(train_inputs)
    losses = []
    for epoch in range(epochs):
        print("--------- EPOCH ",epoch,"-----------")
        X, y = shuffle(train_inputs,train_labels)
        #X,y = train_inputs,train_labels
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
    
    title = "baseline-rnn"
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.title(title)
    plt.savefig(title+'.png')
    plt.close()

    sample_output(pp,model,train_inputs,train_labels)

def test(pp,model,checkpoint,manager,test_inputs,test_labels):
    checkpoint.restore(manager.latest_checkpoint)
    sample_output(pp,model,test_inputs,test_labels)


def sample_output(pp: PreProcessor, model, train_inputs, train_labels):
    n = len(train_inputs)
    rand_idx = randint(0,n)
    sample_input = train_inputs[rand_idx]
    sample_labels = train_labels[rand_idx]
    sample_input_reshaped = tf.reshape(sample_input,(1,-1))
    predicted_output = model(sample_input_reshaped)
    mle_output = tf.argmax(predicted_output,axis=2).numpy().flatten()

    orig_sentence = [pp.idx2word[idx] for idx in sample_input]
    true_tags = [pp.idx2tag[idx] for idx in sample_labels]
    predicted_tags = [pp.idx2tag[idx] for idx in mle_output]

    fancy_print(orig_sentence,predicted_tags,true_tags)

def main():
    pp = PreProcessor()
    #train_folders = ["../training-PHI-Gold-Set1/","../training-PHI-Gold-Set2/"]
    #train_folders = ["../training-PHI-Gold-Set1/"]
    train_folders = ["../Track1-de-indentification/PHI/"]
    X, y = pp.get_data(train_folders)

    model = BaselineModel(pp.vocab_size,pp.tag_size,pp.max_len)
    checkpoint_dir = './checkpoints'
    #checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    #train(pp,model, manager, X,y)
    test(pp,model,checkpoint,manager,X,y)

if __name__ == "__main__":
    main()