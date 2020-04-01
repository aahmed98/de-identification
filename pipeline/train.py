from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
from .visualization import sample_output, loss_plot
import chars2vec as c2v

vanilla = {"baseline-rnn","bi-lstm"}
crf = {"bi-lstm-crf","transformer-crf"}
transformer = {"transformer", "transformer-complex", "transformer-bilstm","transformer-bilstm-complex"}
chars = {"bi-lstm-chars"}

def train(model, train_inputs, train_labels, batch_size = 32,epochs= 10, lr = 0.001, sample_interval = 5, pp = None, manager = None, ckpt = None):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    n = len(train_inputs)
    losses = []
    if manager is not None and ckpt is not None:
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))

    if model.title in chars:
        idx2word = pp.idx2word
        c2v_model = c2v.load_model('eng_50')

    for epoch in range(epochs):  
        print("--------- EPOCH ",epoch,"-----------")
        X, y = shuffle(train_inputs,train_labels)
        epoch_loss = 0
        for i in range(0,n,batch_size):
            input_batch = X[i:i+batch_size]
            output_batch = y[i:i+batch_size]
            if model.title in vanilla:
                with tf.GradientTape() as tape:
                    probs = model(input_batch) # bsz x max_len x tag_size
                    loss = model.loss(probs,output_batch)
            elif model.title in crf:
                with tf.GradientTape() as tape:
                    log_likelihood,_ = model(input_batch,output_batch) # bsz x max_len x tag_size
                    loss = model.loss(log_likelihood)
            elif model.title in chars:
                words = [[idx2word[val] for val in row] for row in input_batch]
                words = np.array(words).flatten() # chars2vec needs list of string
                char_embeddings = c2v_model.vectorize_words(words)
                char_embeddings = np.reshape(char_embeddings, (input_batch.shape[0],input_batch.shape[1], -1))
                with tf.GradientTape() as tape:
                    probs = model(input_batch, char_embeddings) # bsz x max_len x tag_size
                    loss = model.loss(probs,output_batch)
            else: # transformer
                assert model.title in transformer, "invalid model for training"
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

    return losses