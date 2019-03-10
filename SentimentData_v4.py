import h5py
import datetime
import logging
import numpy as np
import pickle
import csv
import json
import re
import os
import nltk

appos = {
    "aren't" : "are not",
    "can't" : "cannot",
    "couldn't" : "could not",
    "didn't" : "did not",
    "doesn't" : "does not",
    "don't": "do not",
    "hadn't" : "had not",
    "hasn't" : "has not",
    "haven't" : "have not",
    "he'd" : "he would",
    "he'll" : "he will",
    "he's" : "he is",
    "i'd" : "I would",
    "i'll" : "I will",
    "i'm" : "I am",
    "isn't" : "is not",
    "it's" : "it is",
    "it'll":"it will",
    "i've" : "I have",
    "let's" : "let us",
    "mightn't" : "might not",
    "mustn't" : "must not",
    "shan't" : "shall not",
    "she'd" : "she would",
    "she'll" : "she will",
    "she's" : "she is",
    "shouldn't" : "should not",
    "should've" : "should have",
    "that's" : "that is",
    "there's" : "there is",
    "they'd" : "they would",
    "they'll" : "they will",
    "they're" : "they are",
    "they've" : "they have",
    "we'd" : "we would",
    "we're" : "we are",
    "weren't" : "were not",
    "we've" : "we have",
    "what'll" : "what will",
    "what're" : "what are",
    "what's" : "what is",
    "what've" : "what have",
    "where's" : "where is",
    "who'd" : "who would",
    "who'll" : "who will",
    "who're" : "who are",
    "who's" : "who is",
    "who've" : "who have",
    "won't" : "will not",
    "wouldn't" : "would not",
    "would've" : "would have",
    "you'd" : "you would",
    "you'll" : "you will",
    "you're" : "you are",
    "you've" : "you have",
    "'re": " are",
    "wasn't": "was not",
    "we'll":" will",
    "must've": "must have",
    "y'all": "you all",
    "it'd ": "it would",
    "ain't": "is not"
}

def make_unicode(inputString):
    """
    Make the string Unicode again
    :param inputString:
    :return:
    """

    try:
        #Python 2
        UNICODE_EXISTS = bool(type(unicode))

        if type(inputString) != unicode:
            inputString = unicode(inputString, 'utf-8')
            return inputString
        else:
            return inputString
    except NameError:
        #python 3
        return str(inputString)


def read_csv_data(csv_file, y_index):
    """
    all X except last column
    :param csv_file:
    :param y_index:
    :return:
    """

    x = []
    y = []

    with open(csv_file) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            for i in range(y_index):
                x.append(row[i])

            # yelp stars to 0-4
            y.append(int(row[y_index])-1)

    X = np.asarray(x)
    Y = np.asarray(y, dtype=int)

    return X, Y


class SentimentData:

    def __init__(self, y_classes, embedding_dim, max_words_len = 100):
        """
        The data class supporting Sentiment model.

        """

        logfilename = "logs/" + str(datetime.date.today()) + ".log"
        logging.basicConfig(filename=logfilename, level=logging.DEBUG, format="%(levelname)s: %(message)s - %(asctime)s")

        try:

            #self.h5File = h5Filename
            self.max_words_len = max_words_len
            self.word_to_vec_map = None

            self.y_classes = y_classes
            self.embedding_dim = embedding_dim

            self.histogram = [
                [0 for i in range(1200)],
                [0 for i in range(1200)],
                [0 for i in range(1200)],
                [0 for i in range(1200)],
                [0 for i in range(1200)]
            ]

        except Exception as ex:
            logging.debug("Error in creating instance: %s", str(ex))

    def yelp_to_csv(self, json_filename, csv_filename, max_number_rows, dev, test):
        """

        :param json_filename:
        :param csv_filename:
        :param max_number_rows: in a csv file
        :param dev: take every __ record
        :param test: take every __ record
        :return:
        """

        rowCount = 1
        fileCount = 1
        training_records = []
        dev_records = []
        test_records = []

        with open(json_filename) as jsonFile:

            for line in jsonFile:

                try:

                    rawData = json.loads(line)

                    # write to Dev file
                    if rowCount % dev == 0:
                        dev_records.append([rawData["text"], rawData["stars"]])

                    # write to test file
                    elif rowCount % test == 0:
                        test_records.append([rawData["text"], rawData["stars"]])

                    # write to train file
                    else:
                        training_records.append([rawData["text"], rawData["stars"]])

                    rowCount += 1

                    # Write to training file
                    if rowCount % max_number_rows == 0:
                        with open(csv_filename + str(fileCount)+ ".csv",'w') as csvTrainFile:
                            wr = csv.writer(csvTrainFile, dialect='excel')
                            wr.writerows(training_records)
                            training_records = []

                        fileCount += 1

                except Exception as e:
                    logging.debug("Error in yelp to csv - error: %s, line: %s" % (str(e), str(line)))

        # write dev file
        with open(csv_filename + ".dev"+ ".csv", 'w') as csvDevFile:
            wr = csv.writer(csvDevFile, dialect='excel')
            wr.writerows(dev_records)

        # write test file
        with open(csv_filename + ".test" + ".csv", 'w') as csvTestFile:
            wr = csv.writer(csvTestFile, dialect='excel')
            wr.writerows(test_records)

        return 0

    def get_words_regex(self, text):
        """
        tokenize the text. Works better!
        :param text:
        :return:
        """
        ctext = make_unicode(text)

        # regex
        regex = r"'?\b[0-9A-Za-z']+\b'?"
        words = re.findall(regex, ctext)

        return words

    def get_words_nltk(self, text):
        """
        tokenize the text.
        :param text:
        :return:
        """
        ctext = make_unicode(text)

        # nltk
        words = nltk.word_tokenize(ctext)

        return words

    def preprocess_words(self, listOfWords):
        """
        Preprocess words
        :param text:
        :return:
        """

        outwords = []

        for w in listOfWords:
            # > lower case
            w2 = w.lower()

            # i'm > i am
            if w2 in appos:
                w2 = appos[w2]
                w2 = w2.split()
                outwords += w2

            # McDonald's > McDonald
            elif "'s" in w2:
                w3 = w2.split("'")
                outwords.append(w3[0])

            # 'hellos' > hellos
            else:
                outwords.append(w2.strip("'"))

        return outwords

    def just_GloVe_words(self, listOfWords):
        """

        :param listOfWords:
        :return:
        """
        w2 = []

        # Could not find these words in GloVe
        for w in listOfWords:

            if (w in self.word_to_vec_map):
                w2.append(w)

        return w2

    def read_glove_vecs(self, glove_file):

        logging.info("Reading from GloVe file: %s " % str(glove_file))

        with open(glove_file, 'r') as f:

            self.word_to_vec_map = {}

            for rawline in f:

                try:

                    # find the first number
                    for i in range(len(rawline)):
                        if (rawline[i].isnumeric() or rawline[i] == "-"):
                            break

                    curr_word = rawline[:i].strip()

                    vectorstr = rawline[i:]
                    vectorlist = vectorstr.strip().split()
                    vectornp = np.array(vectorlist, dtype=np.float64)

                    # make sure all vectors are the same dim
                    vectorout = np.zeros(self.embedding_dim)
                    vectorout[0:vectornp.shape[0]] = vectornp

                    self.word_to_vec_map[curr_word] = vectorout

                except Exception as e:
                    # logging.debug("Error in reading GloVe: %s; line: %s", str(e), rawline)
                    continue

        logging.info("Done from GloVe file: %s " % str(glove_file))
        return 0

    def get_data_properties(self, dir):
        """
        dataproperties = {}
        :param dir:
        :return:
        """

        for (root, dirnames, filenames) in os.walk(dir):

            for f in filenames:

                filenamelist = f.split(".")

                if filenamelist[-1] == "csv":

                    logging.info("Getting data from: %s" % str(f))

                    with open(os.path.join(root, f)) as csvDataFile:
                        csvReader = csv.reader(csvDataFile)

                        for row in csvReader:
                            x = row[0]
                            words = self.get_words_regex(x)
                            #words2 = self.get_words_nltk(x)
                            #logging.info("sentence: %s;\n regex: %s;\n nltk: %s", str(x), str(words1), str(words2))

                            pwords = self.preprocess_words(words)
                            finalwords = self.just_GloVe_words(pwords)

                            starrating = int(row[1]) - 1
                            frequency = len(finalwords)
                            self.histogram[starrating][frequency] += 1

        return 0

    def encode_X_to_indices(self, X):
        """
        Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
        The output shape should be such that it can be given to `Embedding()`.

        Arguments:
        X -- array of sentences (strings), of shape (m, 1)

        Returns:
        X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_words_len)
        """

        m = X.shape[0]  # number of training examples

        # Initialize X_indices as a numpy matrix of zeros and the correct shape
        X_indices = np.zeros((m, self.max_words_len))

        for i in range(m):  # loop over training examples

            # pre processing

            # get words
            words = self.get_words_regex(X[i])
            # preprocess words
            processed_words = self.preprocess_words(words)
            # just give me GloVe
            glove_words = self.just_GloVe_words(processed_words)
            # truncate if too long
            final_words = glove_words[:self.max_words_len]

            # Initialize j to 0
            j = 0

            # Loop over the words of sentence_words
            for w in final_words:
                # Set the (i,j)th entry of X_indices to the index of the correct word.
                X_indices[i, j] = self.word_to_index[w]
                # Increment j to j + 1
                j += 1

        return X_indices

    def encode_X_to_vectors(self, X):
        """
        Converts an array of sentences (strings) into an array of GloVe vectors.

        Arguments:
        X -- array of sentences (strings), of shape (m, 1)

        Returns:
        X_vectors -- array of indices corresponding to words in the sentences from X, of shape (m, max_words_len, emb_dim)
        """

        m = X.shape[0]  # number of training examples

        # Initialize X_vectors as a numpy matrix of zeros and the correct shape
        X_vectors = np.zeros((m, self.max_words_len, self.embedding_dim))

        for i in range(m):  # loop over training examples

            try:

                # pre processing

                # get words
                words = self.get_words_regex(X[i])
                # preprocess words
                processed_words = self.preprocess_words(words)
                # just give me GloVe
                glove_words = self.just_GloVe_words(processed_words)
                # truncate if too long
                final_words = glove_words[:self.max_words_len]

                j = 0

                # Loop over the words of sentence_words
                for w in final_words:
                    # Set the (i,j)th entry of X_indices to the index of the correct word.
                    X_vectors[i, j, :] = self.word_to_vec_map[w]

                    # Increment j to j + 1
                    j+= 1

            except Exception as e:
                    logging.debug("Error in encoding X to vector: %s;\n sentence: %s;\n final words: %s;\n word: %s;\n vectors: %s;",
                                  str(e), str(X[i]), str(final_words), w, str(X_vectors[i, j, :]))

        return X_vectors

    def encode_Y(self, Y):
        """

        :param Y:
        :return:
        """
        Y = np.eye(self.y_classes)[Y.reshape(-1)]
        return Y

    def get_csv_XY(self, csv_file):
        """
        Read and transform data from CSV
        :param csv_file:
        :return:
        """

        X, Y = read_csv_data(csv_file, 1)

        X = self.encode_X_to_vectors(X)
        Y = self.encode_Y(Y)

        return X, Y

    def encode_file(self, csv_file, hdf5_file):
        """

        :param csv_file:
        :param hdf5_file:
        :return:
        """

        X, Y = read_csv_data(csv_file, 1)

        #logging.info("X[1]: %s" % str(X[1]))
        #logging.info("Y[1]: %s" % str(Y[1]))

        X = self.encode_X_to_vectors(X)
        #logging.info("X[1]: %s" % str(X[1]))

        Y = self.encode_Y(Y)
        #logging.info("Y[1]: %s" % str(Y[1]))

        X_h5 = h5py.File(hdf5_file + ".X.h5", 'w')
        xdataset = X_h5.create_dataset("X", data=X)
        X_h5.close()

        Y_h5 = h5py.File(hdf5_file + ".Y.h5", 'w')
        ydataset = Y_h5.create_dataset("Y", data=Y)
        Y_h5.close()

        return X, Y

    def encode_dir(self, csv_dir, hdf5_dir):
        """

        :param csv_file:
        :param hdf5_file:
        :return:
        """

        for (root, dirnames, filenames) in os.walk(csv_dir):

            for file in filenames:

                filenamelist = file.split(".")

                if filenamelist[-1] == "csv":

                    logging.info("Getting data from: %s" % str(file))

                    self.encode_file(os.path.join(root, file), hdf5_dir+'/'+file)

    def decode_Y(self, Y):
        """
        Decode Y
        :param Y:
        :return:
        """
        res = np.zeros(len(Y))

        for i in range(len(Y)):
            res[i] = np.argmax(Y[i]) + 1

        return res

    def save(self, filename):
        """

        :param filename:
        :return:
        """
        try:
            file = open(filename, mode='wb')

            picklableDic = {}

            picklableDic['max_words_len'] = self.histogram
            #picklableDic['word_to_vec_map'] = self.word_to_vec_map
            #picklableDic['word_to_index'] = self.word_to_index
            #picklableDic['index_to_word'] = self.index_to_word

            pickledDic = pickle.dump(picklableDic, file)
            file.close()

            return 0

        except Exception as e:
            logging.debug("Error in saving data: %s", str(e))
            return e

    def load(self, filename):
        """

        :param filename:
        :return:
        """
        try:
            file = open(filename, mode='rb')

            picklableDic = pickle.load(file)

            self.max_words_len = picklableDic['max_words_len']
            #self.word_to_vec_map = picklableDic['word_to_vec_map']
            #self.word_to_index = picklableDic['word_to_index']
            #self.index_to_word = picklableDic['index_to_word']

            file.close()
            return 0

        except Exception as e:
            logging.debug("Error in loading data: %s", str(e))
            return e




if __name__ == '__main__':

    try:

        # max word len for Yelp dataset is 1061
        #d = SentimentData(y_classes=5, max_words_len=100, embedding_dim=50)
        d = SentimentData(y_classes=5, max_words_len=100, embedding_dim=300)


        #initiate the data

        d.yelp_to_csv("data/yelp_json/yelpAll.json", "data/yelp_csv/yelp", max_number_rows=5000, dev=1000, test=1001)
        #d.read_glove_vecs(glove_file="data/glove/glove.840B.300d.txt")
        #d.get_data_properties(dir="data/yelp_csv")
        #logging.info("histogram: %s" % str(d.histogram))
        #d.save("sent.data.v4.sentData")

        #x1,y1, x2, y2 = d.get_csv_XY("train1.csv", "test1.csv")

        #d.yelp_to_csv("data/yelp_json/yelpAll.json", "data/yelp_csv/yelpReview", max_number_rows=200000, dev=1000, test=1001)

        # encode files
        #d.read_glove_vecs(glove_file="data/glove/glove.6B.50d.txt")
        #d.read_glove_vecs(glove_file="data/glove/glove.840B.300d.txt")
        d.encode_dir("data/yelp_csv", "data/yelp_hdf5")
        #d.encode_dir("data/test", "data/test-hdf5")

    except Exception as e:
        logging.debug("Error in data processing: %s", str(e))




