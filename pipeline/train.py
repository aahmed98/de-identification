from sklearn.utils import shuffle
import tensorflow as tf
from .visualization import sample_output, sample_output_CRF, fancy_print, loss_plot

def train_vanilla(model, train_inputs, train_labels, batch_size = 32,epochs= 10, sample_interval = 5, pp = None, manager = None, ckpt = None):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    n = len(train_inputs)
    losses = []
    if manager is not None and ckpt is not None:
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
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

    loss_plot(losses,model.title)
    sample_output(pp,model,train_inputs,train_labels)

def train_CRF(model, train_inputs, train_labels, batch_size = 32,epochs= 10, sample_interval = 5, pp = None, manager = None, ckpt = None):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    n = len(train_inputs)
    losses = []
    if manager is not None and ckpt is not None:
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
    for epoch in range(epochs):
        print("--------- EPOCH ",epoch,"-----------")
        X, y = shuffle(train_inputs,train_labels)
        epoch_loss = 0
        for i in range(0,n,batch_size):
            input_batch = X[i:i+batch_size]
            output_batch = y[i:i+batch_size]
            with tf.GradientTape() as tape:
                log_likelihood,_ = model(input_batch,output_batch) # bsz x max_len x tag_size
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
    
    loss_plot(losses,model.title)
    sample_output_CRF(pp,model,train_inputs,train_labels)

def train_transformer(model, train_inputs, train_labels, batch_size = 32,epochs= 10, sample_interval = 5, pp = None, manager = None, ckpt = None):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    n = len(train_inputs)
    losses = []
    if manager is not None and ckpt is not None:
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
    for epoch in range(epochs):  
        print("--------- EPOCH ",epoch,"-----------")
        X, y = shuffle(train_inputs,train_labels)
        epoch_loss = 0
        for i in range(0,n,batch_size):
            input_batch = X[i:i+batch_size]
            output_batch = y[i:i+batch_size]
            with tf.GradientTape() as tape:
                probs = model(input_batch,output_batch) # bsz x max_len x tag_size
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

    loss_plot(losses,model.title)
    sample_output(pp,model,train_inputs,train_labels)