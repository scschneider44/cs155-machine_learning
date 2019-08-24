# Solution set for CS 155 Set 6, 2017
# Authors: Suraj Nair, Sid Murching

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from P3CHelpers import *
import sys

def get_word_repr(word_to_index, word):
    """
    Returns one-hot-encoded feature representation of the specified word given
    a dictionary mapping words to their one-hot-encoded index.

    Arguments:
        word_to_index: Dictionary mapping words to their corresponding index
                       in a one-hot-encoded representation of our corpus.

        word:          String containing word whose feature representation we wish to compute.

    Returns:
        feature_representation:     Feature representation of the passed-in word.
    """
    unique_words = word_to_index.keys()
    # Return a vector that's zero everywhere besides the index corresponding to <word>
    feature_representation = np.zeros(len(unique_words))
    feature_representation[word_to_index[word]] = 1
    return feature_representation    

def generate_traindata(word_list, word_to_index, window_size=4):
    """
    Generates training data for Skipgram model.

    Arguments:
        word_list:     Sequential list of words (strings).
        word_to_index: Dictionary mapping words to their corresponding index
                       in a one-hot-encoded representation of our corpus.

        window_size:   Size of Skipgram window.
                       (use the default value when running your code).

    Returns:
        (trainX, trainY):     A pair of matrices (trainX, trainY) containing training 
                              points (one-hot-encoded vectors representing individual words) and 
                              their corresponding labels (also one-hot-encoded vectors representing words).

                              For each index i, trainX[i] should correspond to a word in
                              <word_list>, and trainY[i] should correspond to one of the words within
                              a window of size <window_size> of trainX[i].
    """
    do = ""
    trainX = []
    trainY = []
    idx = [-4, -3, -2, -1, 1, 2, 3, 4]
    # TODO: Implement this function, populating trainX and trainY
    for i in range(len(word_list)):
        for j in range(8):
            if (i + idx[j]) < 0 or (i + idx[j]) > len(word_list) - 1:
                do = "nothing"
            else :
                trainX.append(get_word_repr(word_to_index, word_list[i]))
                trainY.append(get_word_repr(word_to_index, word_list[i + idx[j]]))
    return (np.array(trainX), np.array(trainY))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python P3C.py <path_to_textfile> <num_latent_factors>")
        sys.exit(1)

    filename = sys.argv[1]
    num_latent_factors = int(sys.argv[2])
    # Load in a list of words from the specified file; remove non-alphanumeric characters
    # and make all chars lowercase.
    sample_text = load_word_list(filename)

    # Create dictionary mapping unique words to their one-hot-encoded index
    word_to_index = generate_onehot_dict(sample_text)
    # Create training data using default window size
    trainX, trainY = generate_traindata(sample_text, word_to_index)
    # TODO: 1) Create and train model in Keras.      
    # vocab_size = number of unique words in our text file. Will be useful when adding layers
    # to your neural network
    vocab_size = len(word_to_index)
    model = Sequential()
    model.add(Dense(10, input_shape = (vocab_size,)))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='Adam')
    fit = model.fit(trainX, trainY, epochs = 10)
    # TODO: 2) Extract weights for hidden layer, set <weights> variable below
    weights = None
    weights = model.layers[0].get_weights()[0]
    print(weights.shape)
    print(model.layers[1].get_weights()[0].shape)
    #weights = None

    # Find and print most similar pairs
    similar_pairs = most_similar_pairs(weights, word_to_index)
    for pair in similar_pairs[:30]:
        print(pair)
