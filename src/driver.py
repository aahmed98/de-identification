from preprocess import PreProcessor
from baseline import BaselineModel
from random import randint
from sklearn.utils import shuffle
import tensorflow as tf
#tf.enable_eager_execution()
import matplotlib.pyplot as plt

def fancy_print(input_sentence, predictions, labels):
    print("{:15} {:5}: ({})".format("Word", "Pred", "True"))
    print("="*30)
    for w, true, pred in zip(input_sentence, predictions, labels):
        if w != "__PAD__":
            print("{:15}:{:5} ({})".format(w, true, pred))

def train(pp, model, train_inputs, train_labels, batch_size):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    n = len(train_inputs)
    losses = []
    for epoch in range(100):
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
            # if i % (batch_size*3) == 0:
            #     print("Loss after batch ",int(i/batch_size), ": ",loss)
        #sample_output(pp,model,train_inputs,train_labels)
        print("LOSS: ",epoch_loss)
        losses.append(epoch_loss)
    
    title = "baseline-rnn"
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.title(title)
    plt.savefig(title+'.png')
    plt.close()

    for _ in range(5):
        sample_output(pp,model,train_inputs,train_labels)

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
    X, y = pp.get_data("../Track1-de-indentification/PHI/")
    model = BaselineModel(pp.vocab_size,pp.tag_size,pp.max_len)
    train(pp,model,X,y,32)

if __name__ == "__main__":
    main()