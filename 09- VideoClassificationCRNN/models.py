import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, SimpleRNN, GRU, LSTM, Dense, Flatten, TimeDistributed


class Models():

      def __init__(self, input_shape, max_seq_len):
        self.frame_height = input_shape[0]
        self.frame_width = input_shape[1]
        self.max_seq_len = max_seq_len

        self.frame_input = tf.keras.Input((None, self.frame_height, self.frame_width, 3))
        self.mask_input = tf.keras.Input((self.max_seq_len,), dtype="bool")


      # Backbone Model(feature extractor)
      def backbone_model(self):

        feature_extractor = tf.keras.applications.ResNet50V2(
          include_top=False,
          weights="imagenet",
          input_tensor=None,
          input_shape=(self.frame_height, self.frame_width, 3),
          pooling='avg',
          classes=2,
          classifier_activation="softmax")
        
        for layer in feature_extractor.layers: # Freeze the layers
          layer.trainable=False

        return feature_extractor


      # RNN Model
      def RNN_model(self):

          features = TimeDistributed(self.backbone_model())(self.frame_input)
          flatten = TimeDistributed(Flatten())(features)
          droupout = Dropout(0.4)(flatten)

          rnn = SimpleRNN(64)(droupout, mask=self.mask_input)                                       
          output = Dense(2, activation="softmax")(rnn)

          rnn_model = tf.keras.Model([self.frame_input, self.mask_input], output)

          return rnn_model



      # GRU Model
      def GRU_model(self):

          features = TimeDistributed(self.backbone_model())(self.frame_input)
          flatten = TimeDistributed(Flatten())(features)
          droupout = Dropout(0.4)(flatten)

          gru = GRU(64)(droupout, mask=self.mask_input)                            
          output = Dense(2, activation="softmax")(gru)

          gru_model = tf.keras.Model([self.frame_input, self.mask_input], output)

          return gru_model


      # LSTM Model
      def LSTM_model(self):

          features = TimeDistributed(self.backbone_model())(self.frame_input)
          flatten = TimeDistributed(Flatten())(features)
          droupout = Dropout(0.4)(flatten)

          lstm = LSTM(64)(droupout, mask=self.mask_input)                                       
          output = Dense(2, activation="softmax")(lstm)

          lstm_model = tf.keras.Model([self.frame_input, self.mask_input], output)

          return lstm_model