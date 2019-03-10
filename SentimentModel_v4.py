#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import keras as k
import numpy as np
import h5py
import os
import logging
import datetime
import time
import warnings
warnings.filterwarnings('ignore')

import SentimentData_v4


def get_hdf5_data(hdf5_file, dataset, fromIndex, toIndex):
    """
    Slice of HDF5 data 
     
    :param hdf5_file: 
    :param dataset: 
    :param fromIndex: 
    :param toIndex: 
    :return:
    
    """

    with h5py.File(hdf5_file, 'r') as h5f:
        data = h5f[dataset]
        return data[fromIndex:toIndex, :]


class Sentiment:

    def __init__(self, input_shape):
        """
        Sentiment Analysis with GloVe + LSTM
        
        :param max_len: 
        """

        logfilename = "logs/" + str(datetime.date.today()) + ".log"
        logging.basicConfig(filename=logfilename, level=logging.DEBUG,
                            format="%(levelname)s: %(message)s - %(asctime)s")

        self.X_train = None
        self.Y_train = None

        self.X_dev = None
        self.Y_dev = None

        self.X_test = None
        self.Y_test = None

        self.model = None
        self.scores = None
        self.input_shape = input_shape

        #self.read_glove_vecs(glove_file)

    def create_model_GloVe_average(self):
        """
        The sentiment analysis model's graph.

        :return: model -- a model instance in Keras
        """

        input_layer = k.layers.Input(shape=(self.input_shape[0],self.input_shape[1],), dtype='float32')


        X = k.layers.Dense(units=5, name="dense")(input_layer)
        X = k.layers.AveragePooling1D(pool_size=100, strides=None, padding='same', name="average")(X)
        X = k.layers.Flatten(name="flatten")(X)
        X = k.layers.Activation('softmax', name="softmax")(X)

        self.model = k.models.Model(input_layer, X)

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return 0

    def create_model_LSTM(self):
        """
        The sentiment analysis model's graph.

        :return: model -- a model instance in Keras
        """

        input_layer = k.layers.Input(shape=(self.input_shape[0], self.input_shape[1],), dtype='float32')

        X = k.layers.LSTM(units=128, return_sequences=True)(input_layer)
        X = k.layers.Dropout(rate=0.5)(X)
        X = k.layers.LSTM(units=128, return_sequences=False)(X)
        X = k.layers.Dropout(rate=0.5)(X)
        X = k.layers.Dense(units=5)(X)
        X = k.layers.Activation('softmax')(X)

        self.model = k.models.Model(input_layer, X)

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return 0

    def create_model_BiLSTM(self):
        """
        The sentiment analysis model's graph.

        :return: model -- a model instance in Keras
        """

        input_layer = k.layers.Input(shape=(self.input_shape[0], self.input_shape[1],), dtype='float32')

        X = k.layers.Bidirectional(k.layers.LSTM(units=512, return_sequences=True, name="BiLSTM_1"))(input_layer)
        X = k.layers.Dropout(rate=0.5)(X)
        X = k.layers.Bidirectional(k.layers.LSTM(units=256, return_sequences=True, name="BiLSTM_2"))(X)
        X = k.layers.Dropout(rate=0.5)(X)
        X = k.layers.Bidirectional(k.layers.LSTM(units=128, return_sequences=True, name="BiLSTM_3"))(X)
        X = k.layers.Dropout(rate=0.5)(X)
        X = k.layers.LSTM(units=64, return_sequences=False, name="LSTM_4_final")(X)
        X = k.layers.Dropout(rate=0.5)(X)
        X = k.layers.Dense(units=5, name="Dense_final")(X)
        X = k.layers.Activation('softmax')(X)

        self.model = k.models.Model(input_layer, X)

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return 0

    def create_model_DPCNN(self):
        """
        Creating the DPCNN model.
        :return:
        """
        # hyper parameters
        filter_nr = 64
        filter_size = 3
        max_pool_size = 3
        max_pool_strides = 2
        dense_nr = 256
        spatial_dropout = 0.2
        dense_dropout = 0.5
        train_embed = False
        conv_kern_reg = k.regularizers.l2(0.00001)
        conv_bias_reg = k.regularizers.l2(0.00001)

        # The model - here we go

        # Embedding layer is done in GloVe
        input_layer = k.layers.Input(shape=self.input_shape)
        emb_comment = k.layers.SpatialDropout1D(spatial_dropout)(input_layer)

        # Block 1
        block1 = k.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(emb_comment)
        block1 = k.layers.BatchNormalization()(block1)
        block1 = k.layers.PReLU()(block1)
        block1 = k.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1)
        block1 = k.layers.BatchNormalization()(block1)
        block1 = k.layers.PReLU()(block1)

        # we pass embedded comment through conv1d with filter size 1 because it needs to have the same shape as block output
        resize_emb = k.layers.Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear',
                            kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(emb_comment)
        resize_emb = k.layers.PReLU()(resize_emb)

        block1_output = k.layers.add([block1, resize_emb])
        block1_output = k.layers.MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)

        # Block 2
        block2 = k.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1_output)
        block2 = k.layers.BatchNormalization()(block2)
        block2 = k.layers.PReLU()(block2)
        block2 = k.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2)
        block2 = k.layers.BatchNormalization()(block2)
        block2 = k.layers.PReLU()(block2)

        block2_output = k.layers.add([block2, block1_output])
        block2_output = k.layers.MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block2_output)

        block3 = k.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2_output)
        block3 = k.layers.BatchNormalization()(block3)
        block3 = k.layers.PReLU()(block3)
        block3 = k.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3)
        block3 = k.layers.BatchNormalization()(block3)
        block3 = k.layers.PReLU()(block3)

        block3_output = k.layers.add([block3, block2_output])
        block3_output = k.layers.MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block3_output)

        block4 = k.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3_output)
        block4 = k.layers.BatchNormalization()(block4)
        block4 = k.layers.PReLU()(block4)
        block4 = k.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4)
        block4 = k.layers.BatchNormalization()(block4)
        block4 = k.layers.PReLU()(block4)

        block4_output = k.layers.add([block4, block3_output])
        block4_output = k.layers.MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block4_output)

        block5 = k.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4_output)
        block5 = k.layers.BatchNormalization()(block5)
        block5 = k.layers.PReLU()(block5)
        block5 = k.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5)
        block5 = k.layers.BatchNormalization()(block5)
        block5 = k.layers.PReLU()(block5)

        block5_output = k.layers.add([block5, block4_output])
        block5_output = k.layers.MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block5_output)

        '''
        block6 = k.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5_output)
        block6 = k.layers.BatchNormalization()(block6)
        block6 = k.layers.PReLU()(block6)
        block6 = k.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block6)
        block6 = k.layers.BatchNormalization()(block6)
        block6 = k.layers.PReLU()(block6)

        block6_output = k.layers.add([block6, block5_output])
        block6_output = k.layers.MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block6_output)

        block7 = k.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block6_output)
        block7 = k.layers.BatchNormalization()(block7)
        block7 = k.layers.PReLU()(block7)
        block7 = k.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear',
                        kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block7)
        block7 = k.layers.BatchNormalization()(block7)
        block7 = k.layers.PReLU()(block7)

        block7_output = k.layers.add([block7, block6_output])
        '''

        output = k.layers.GlobalMaxPooling1D()(block5_output)
        output = k.layers.Dense(dense_nr, activation='linear')(output)
        output = k.layers.BatchNormalization()(output)
        output = k.layers.PReLU()(output)
        output = k.layers.Dropout(dense_dropout)(output)
        output = k.layers.Dense(5, activation='softmax')(output)

        self.model = k.Model(input_layer, output)

        self.model.compile(loss='binary_crossentropy',
                      optimizer=k.optimizers.Adam(),
                      metrics=['accuracy'])

        return 0


    def create_model_ULMFiT(self):
        """
        The sentiment analysis model's graph.

        :return: model -- a model instance in Keras
        """

        input_layer = k.layers.Input(shape=(self.input_shape[0], self.input_shape[1],), dtype='float32')

        X = k.layers.LSTM(units=1150, return_sequences=True, name="LSTM_1")(input_layer)
        X = k.layers.LSTM(units=1150, return_sequences=True, name="LSTM_2")(X)
        X = k.layers.LSTM(units=1150, return_sequences=False, name="LSTM_3")(X)
        X = k.layers.Dense(units=5, name="Dense_final")(X)
        X = k.layers.Activation('softmax')(X)

        self.model = k.models.Model(input_layer, X)

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return 0


    def get_training_data(self, hdf5_X_file, hdf5_Y_file):
        """
        Import training X,Y from HDF5 files
        :param hdf5_X_file:
        :param hdf5_Y_file:
        :return:
        """
        self.X_train = get_hdf5_data(hdf5_X_file,"X", 0, -1)
        self.Y_train = get_hdf5_data(hdf5_Y_file, "Y", 0, -1)

    def train_model(self, epochs, batch_size=128):
        """
        Train model on the existing X,Y
        :param epochs:
        :param batch_size:
        :return:
        """

        #logging.info("X_train Shape: %s;\n Y_train shape: %s", str(self.X_train.shape), str(self.Y_train.shape))
        tensorboard_callback = k.callbacks.TensorBoard(log_dir="logs/tb/{}".format(time.time()))

        self.model.fit(self.X_train,
                       self.Y_train,
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=True,
                       validation_split=0.0,
                       callbacks=[tensorboard_callback],
                       verbose=1)

    def train_from_dir(self, X_dir, Y_dir, epochs, save_file, batch_size=128, quit_threshold=0.005):
        """
        Train the models on all the files in the directory
        :param dir:
        :param filesEpochs:
        :param dataPointEpochs:
        :param batch_size:
        :return:
        """

        # number of epochs to go through all the files
        for i in range(epochs):

            for (root, dirnames, filenames) in os.walk(X_dir):

                for f in filenames:

                    filenamelist = f.split(".")

                    if filenamelist[-1] == "h5":

                        logging.info("Training on: %s" % str(f))

                        # find X and Y filenames
                        Xfile = os.path.join(root, f)
                        Yfile = Y_dir+"/"+f[0:-4]+"Y.h5"

                        # get the data from files
                        self.get_training_data(Xfile, Yfile)

                        # train the mode
                        self.train_model(epochs=1, batch_size=batch_size)

                        #save the model
                        self.save_model(save_file)

                        #logging.info("Trained on %s: " % str(f))

            '''
            # the model is trained on all the files ANOTHER time
            if self.scores:
                prev_loss = self.scores[0]
            else:
                prev_loss = 1000000000

            # evaluate the model
            self.evaluate_model()
            current_loss = self.scores[0]

            logging.info("loss: %s" % str(current_loss))

            # we have reached a stable state
            if prev_loss - current_loss < quit_threshold:
                break
            '''

        logging.info("Done with directory %s: " % str(X_dir))
        return 0

    def get_dev_data(self, hdf5_dev_X_file, hdf5_dev_Y_file):
        """
        Import training X,Y from HDF5 files
        :param hdf5_X_file:
        :param hdf5_Y_file:
        :return:
        """
        self.X_dev = get_hdf5_data(hdf5_dev_X_file,"X", 0, -1)
        self.Y_dev = get_hdf5_data(hdf5_dev_Y_file, "Y", 0, -1)

    def evaluate_model(self):
        # Evaluation on the test set
        self.scores = self.model.evaluate(self.X_dev, self.Y_dev, verbose=0)
        logging.info("metrics: %s; scores: %s" , str(self.model.metrics_names), str(self.scores))
        print("metrics: %s; scores: %s" % (str(self.model.metrics_names), str(self.scores)))

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model = k.models.load_model(filename)

    def get_sentiment(self, X_test):
        'Returns the sentiment for a block of text. between 1-5.'

        try:

            #X_test_indices = self.sentences_to_indices(X_test)
            pred = self.model.predict(X_test)
            return pred

        except Exception as e:
            sys.stderr.write("Sentiment error: %s.\n" % str(e))
            return -2


if __name__ == '__main__':

    # create obj and initialize
    sentObj = Sentiment(input_shape=(100,300))

    #sentObj.read_glove_vecs(glove_file="data/glove/glove.6B.50d.txt", vector_len=50)
    #sentObj.read_glove_vecs(glove_file="data/glove/glove.840B.300d.txt", vector_len=300)

    # Create Model
    '''
    #sentObj.create_model_GloVe_average()
    sentObj.create_model_DPCNN()
    sentObj.model.summary()
    sentObj.save_model("sentimentModel_DPCNN.keras")
    '''

    # Training
    '''
    sentObj.load_model("sentimentModel_DPCNN.keras")
    sentObj.model.summary()
    #sentObj.get_dev_data("data/yelp_hdf5_devtest/yelp.dev.csv.X.h5", "data/yelp_hdf5_devtest/yelp.dev.csv.Y.h5")

    # on p3.16xl 736s 4ms/step (14x) - on laptop 10910s 55ms/step
    sentObj.train_from_dir(X_dir="data/yelp_hdf5_X",
                           Y_dir="data/yelp_hdf5_Y",
                           epochs=100,
                           save_file="sentimentModel_DPCNN.keras",
                           batch_size=3000)

    sentObj.save_model("sentimentModel_DPCNN.keras")
    '''

    # Evaluate

    sentObj.load_model("models/DPCNN.keras")
    sentObj.model.summary()
    sentObj.get_dev_data("data/yelp_hdf5_devtest/yelp.dev.csv.X.h5",
                         "data/yelp_hdf5_devtest/yelp.dev.csv.Y.h5")
    sentObj.evaluate_model()
    '''

    # Inference
    '''
    p = ["This is a rad company.",
         "This is not a very good day.",
         "yeah right",
         "I don't like it",
         "super nice",
         "very bad company"
        ]
    a = np.asarray(p)

    d = SentimentData_v4.SentimentData(y_classes=5, max_words_len=100, embedding_dim=300)
    d.read_glove_vecs(glove_file="data/glove/glove.840B.300d.txt")
    x = d.encode_X_to_vectors(a)

    sentObj.load_model("models/DPCNN.keras")
    sent = sentObj.get_sentiment(x)
    res = d.decode_Y(sent)
    logging.info("Sentiment: %s " % str(res))
    print("Sentiment: %s " % str(res))

