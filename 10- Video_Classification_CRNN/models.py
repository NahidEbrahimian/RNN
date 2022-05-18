import tensorflow as tf
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dropout, SimpleRNN, GRU, LSTM, Dense, Flatten, TimeDistributed
from tensorflow.keras.activations import relu
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

class Models():

      def __init__(self, input_shape, max_seq_len):
        self.frame_height = input_shape[0]
        self.frame_width = input_shape[1]
        self.max_seq_len = max_seq_len

        self.frame_input = tf.keras.Input((None, self.frame_height, self.frame_width, 3))
        self.mask_input = tf.keras.Input((self.max_seq_len,), dtype="bool")
        # self.frame_input = tf.keras.Input((None, 20, 2048))
        # self.mask_input = tf.keras.Input((20,), dtype="bool")

      # Backbone Model(feature extractor)
      def backbone_model(self):

      #   feature_extractor = tf.keras.applications.ResNet50V2(
      #     include_top=False,
      #     weights="imagenet",
      #     input_tensor=None,
      #     input_shape=(None, self.frame_height, self.frame_width, 3),
      #     pooling='avg',
      #     classes=2,
      #     classifier_activation="softmax")
        
      #   for layer in feature_extractor.layers[:-5]: # Freeze the layers
      #     layer.trainable=False

        num_filters = 16
        conv11 = TimeDistributed(Conv2D(filters=num_filters, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='HeNormal', kernel_regularizer=l2(1e-4)))(self.frame_input)
        conv11 = TimeDistributed(BatchNormalization())(conv11)
        conv11 = relu(conv11)
        conv11 = TimeDistributed(Conv2D(filters=num_filters, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='HeNormal', kernel_regularizer=l2(1e-4)))(conv11)
        conv11 = TimeDistributed(BatchNormalization())(conv11)
        conv11 = relu(conv11)
        conv11 = TimeDistributed(Conv2D(filters=num_filters, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='HeNormal', kernel_regularizer=l2(1e-4)))(conv11)
        conv11 = TimeDistributed(BatchNormalization())(conv11)
        conv11 = relu(conv11)
        conv11 = TimeDistributed(Conv2D(filters=num_filters, kernel_size=(3, 3), strides=2, padding='same', kernel_initializer='HeNormal'))(conv11)

        num_filters = num_filters * 2
        conv12 = TimeDistributed(Conv2D(filters=num_filters, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='HeNormal', kernel_regularizer=l2(1e-4)))(conv11)
        conv12 = TimeDistributed(BatchNormalization())(conv12)
        conv12 = relu(conv12)
        conv12 = TimeDistributed(Conv2D(filters=num_filters, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='HeNormal', kernel_regularizer=l2(1e-4)))(conv12)
        conv12 = TimeDistributed(BatchNormalization())(conv12)
        conv12 = relu(conv12)
        conv12 = TimeDistributed(Conv2D(filters=num_filters, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='HeNormal', kernel_regularizer=l2(1e-4)))(conv12)
        conv12 = TimeDistributed(BatchNormalization())(conv12)
        conv12 = relu(conv12)
        conv12 = TimeDistributed(Conv2D(filters=num_filters, kernel_size=(3, 3), strides=2, padding='same', kernel_initializer='HeNormal'))(conv12)

        num_filters = num_filters * 2
        conv13 = TimeDistributed(Conv2D(filters=num_filters, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='HeNormal', kernel_regularizer=l2(1e-4)))(conv12)
        conv13 = TimeDistributed(BatchNormalization())(conv13)
        conv13 = relu(conv13)
        conv13 = TimeDistributed(Conv2D(filters=num_filters, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='HeNormal', kernel_regularizer=l2(1e-4)))(conv13)
        conv13 = TimeDistributed(BatchNormalization())(conv13)
        conv13 = relu(conv13)
        conv13 = TimeDistributed(Conv2D(filters=num_filters, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='HeNormal', kernel_regularizer=l2(1e-4)))(conv13)
        conv13 = TimeDistributed(BatchNormalization())(conv13)
        conv13 = relu(conv13)
        conv13 = TimeDistributed(Conv2D(filters=num_filters, kernel_size=(3, 3), strides=2, padding='same', kernel_initializer='HeNormal'))(conv13)

        feature_extractor = Model(self.frame_input, conv13)

        return feature_extractor


      # RNN Model
      def RNN_model(self):

          features = self.backbone_model()(self.frame_input)
          flatten = TimeDistributed(Flatten())(features)
          droupout = Dropout(0.4)(flatten)

          # rnn = SimpleRNN(64)(self.frame_input, mask=self.mask_input) 
          rnn = SimpleRNN(64)(droupout, mask=self.mask_input)                                       
          output = Dense(10, activation="softmax")(rnn)

          rnn_model = tf.keras.Model([self.frame_input, self.mask_input], output)

          return rnn_model



      # GRU Model
      def GRU_model(self):

          features = self.backbone_model()(self.frame_input)
          flatten = TimeDistributed(Flatten())(features)
          droupout = Dropout(0.4)(flatten)

          gru = GRU(64)(droupout, mask=self.mask_input)                            
          output = Dense(2, activation="softmax")(gru)

          gru_model = tf.keras.Model([self.frame_input, self.mask_input], output)

          return gru_model


      # LSTM Model
      def LSTM_model(self):

          features = self.backbone_model()(self.frame_input)
          flatten = TimeDistributed(Flatten())(features)
          droupout = Dropout(0.4)(flatten)

          lstm = LSTM(64)(droupout, mask=self.mask_input)                                       
          output = Dense(2, activation="softmax")(lstm)

          lstm_model = tf.keras.Model([self.frame_input, self.mask_input], output)

          return lstm_model
