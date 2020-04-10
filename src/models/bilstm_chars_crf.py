import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import chars2vec as c2v

class BiLSTM_Chars_CRF(tf.keras.Model):
    """
    BiLSTM seq2seq.
    """
    def __init__(self,vocab_size,tag_size,max_len,idx2word: dict):
        super(BiLSTM_Chars_CRF,self).__init__()
        self.vocab_size =  vocab_size + 1 #add 1 because of weird problem with embedding lookup. only happens on large data. CPU/GPU related I think
        self.tag_size = tag_size
        self.max_len = max_len
        self.embedding_size = 64
        self.rnn_size = 128
        self.title = "bi-lstm-chars-crf"
        self.idx2word = idx2word
        self.c2v_model = c2v.load_model('eng_50')

        self.transition_params = tf.random.uniform((self.tag_size,self.tag_size)) # for CRF

        self.E = tf.Variable(tf.random.normal([self.vocab_size,self.embedding_size],stddev = 0.1, dtype= tf.float32))
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_size, return_sequences = True)) # automatically sets backward layer to be identical to forward layer
        self.d1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.tag_size)) # softmax over tags for each word

    @tf.function(experimental_relax_shapes=True)
    def call(self,inputs, char_embeddings, labels):
        """
        Inputs: (batch_size, max_len)
        Output: (batch_size, max_len, tag_size) 
        """
        embeddings = tf.nn.embedding_lookup(self.E,inputs) # (batch_size, max_len, embedding_size)
        embeddings = tf.concat([embeddings,char_embeddings],axis=2) # (batch_size, max_len, embedding_size + 50)
        # print(embeddings.shape)
        outputs = self.bi_lstm(embeddings) # (batch_size, max_len, 2*rnn_size)
        logits = self.d1(outputs) # (batch_size, max_len, tag_size)

        mask = tf.cast(tf.not_equal(labels, 0), tf.float32)
        true_lengths = tf.reduce_sum(mask,axis = 1) # likelihood should ignore masks
        log_likelihood, transition = tfa.text.crf_log_likelihood(logits,labels,true_lengths,self.transition_params)
        return log_likelihood, transition

    def loss(self,log_likelihood):
        return tf.reduce_mean(-1*log_likelihood)

    def predict(self,inputs, words = None):
        np_input = inputs.numpy()
        if words is None:
            words = [[self.idx2word[val] for val in row] for row in np_input]
            words = np.array(words).flatten() # chars2vec needs list of string
        else: 
            words = words.flatten()
            words = words.astype('str')

        char_embeddings = self.c2v_model.vectorize_words(words)
        char_embeddings = tf.reshape(char_embeddings, [1,-1,50])

        embeddings = tf.nn.embedding_lookup(self.E,inputs) # (batch_size, max_len, embedding_size)
        embeddings = tf.concat([embeddings,char_embeddings],axis=2) # (batch_size, max_len, embedding_size + 50)
        # print(embeddings.shape)
        outputs = self.bi_lstm(embeddings) # (batch_size, max_len, 2*rnn_size)
        logits = self.d1(outputs) # (batch_size, max_len, tag_size)

        pre_seqs = []
        for score in logits:
            pre_seq, _ = tfa.text.viterbi_decode(score[:], self.transition_params)
            pre_seqs.append(pre_seq)
        return np.array(pre_seqs).flatten()

