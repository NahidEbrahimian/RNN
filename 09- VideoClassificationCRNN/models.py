import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, SimpleRNN, GRU, LSTM, Dense, Flatten, TimeDistributed

# RNN Model

def RNN_model(max_seq_len):

    frame_width = 112
    frame_height = 112

    frame_input = tf.keras.Input((None, frame_height, frame_width, 3))
    mask_input = tf.keras.Input((max_seq_len,), dtype="bool")

    conv = TimeDistributed(Conv2D(16, (3, 3), activation="relu", padding="same"))(frame_input)
    maxpool = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv)
    flatten = TimeDistributed(Flatten())(maxpool)
                                        
    rnn = SimpleRNN(64, return_sequences=True)(flatten, mask=mask_input)  
    rnn = SimpleRNN(32)(rnn)     
    output = Dense(2, activation="softmax")(rnn)

    rnn_model = tf.keras.Model([frame_input, mask_input], output)

    return rnn_model



  # GRU Model

def GRU_model(max_seq_len):

    frame_width = 112
    frame_height = 112

    frame_input = tf.keras.Input((None, frame_height, frame_width, 3))
    mask_input = tf.keras.Input((max_seq_len,), dtype="bool")

    conv = TimeDistributed(Conv2D(16, (3, 3), activation="relu", padding="same"))(frame_input)
    maxpool = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv)
    flatten = TimeDistributed(Flatten())(maxpool)
                                        
    gru = GRU(64, return_sequences=True)(flatten, mask=mask_input)  
    gru = GRU(32)(gru)     
    output = Dense(2, activation="softmax")(gru)

    gru_model = tf.keras.Model([frame_input, mask_input], output)

    return gru_model


# LSTM Model

def LSTM_model(max_seq_len):

    frame_width = 112
    frame_height = 112

    frame_input = tf.keras.Input((None, frame_height, frame_width, 3))
    mask_input = tf.keras.Input((max_seq_len,), dtype="bool")

    conv = TimeDistributed(Conv2D(16, (3, 3), activation="relu", padding="same"))(frame_input)
    maxpool = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv)
    flatten = TimeDistributed(Flatten())(maxpool)
                                        
    lstm = GRU(64, return_sequences=True)(flatten, mask=mask_input)  
    lstm = GRU(32)(lstm)     
    output = Dense(2, activation="softmax")(lstm)

    lstm_model = tf.keras.Model([frame_input, mask_input], output)

    return lstm_model