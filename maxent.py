# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
import collections
from nltk.corpus import stopwords
from nltk import word_tokenize

import numpy as np
from scipy.special import logsumexp


class MaxEnt(Classifier):

    # added
    def __init__(self):
        # class constructor of parent class so as to no override it
        # Classifier.__init__(self, None)
        # specific parameters of this MaxEnt subclass
        self.training_doc_list = None
        self.devset_list = None
        self.weights_matrix = None
        self.temp_matrix = None
        self.word_set_features = None
        self.label_set = None
        self.feature_set = None

    def get_model(self):
        return None

    def set_model(self, model):
        pass

    model = property(get_model, set_model)

    def train(self, instances, dev_instances=None):
        """Construct a statistical model from labeled instances."""
        self.train_sgd(instances, dev_instances, 0.0001, 30)

    def train_sgd(self, train_instances, dev_instances, learning_rate, batch_size):
        """Train MaxEnt model with Mini-batch Stochastic Gradient"""

        # creates the feature and label sets and stores them as attributes
        self.create_feature_set(train_instances)
        self.create_label_set(train_instances[0])
        # creates matrix to store the weights
        self.weights_matrix = np.zeros([len(self.label_set), len(self.feature_set)])

        minibatch_index = 0
        converged = False
        # arbitrary number to decide when to declare convergence
        chances_left = 10
        old_accuracy = 0

        # SGD loop
        while not converged:
            # get a new minibatch with each iteration
            minibatch = train_instances[minibatch_index : minibatch_index+batch_size]
            minibatch_index += batch_size

            if minibatch_index > len(train_instances):
                minibatch_index = 0

            # matrices to store the observed and expected counts of the derivative of the log likelihood
            observed_matrix = np.zeros_like(self.weights_matrix)
            expected_matrix = np.zeros_like(self.weights_matrix)

            likelihood = 0

            # for each doc in the minibatch
            for doc in minibatch:
                # get the index of the true label for the given doc in the matrices
                doc_label_index = self.label_index_dict[doc.label]

                # select some features or other depending on the kind of documents we are classifying
                if "positive" in self.label_set:
                    doc_features = doc.features().intersection(self.feature_set)
                else:
                    doc_features = doc.features()

                # creates a sparse feature vector containing the indices of the features that the given doc contains
                sparse_feature_vector = [self.feature_index_dict[feature] for feature in doc_features]

                # add one to the respective cell of the observed counts matrix
                observed_matrix[doc_label_index, sparse_feature_vector] += 1

                # the exponent of the numerator for a specific label of a doc: self.weights_matrix[doc.label,spares
                #self.weights_matrix[doc.label, sparse_feature_vector].sum()

                # the denominator (sum(1) does the sum among the elements of the same row)
                #log_denominator = logsumexp(self.weights_matrix[:, sparse_feature_vector].sum(1))

                # probs_column is a column vector of the size of the label set
                # where each of the values are the weights corresponding to the features of the document
                # This is a temporary value that will be eventually substituted by the final probability of that label
                probs_column = self.weights_matrix[:, sparse_feature_vector].sum(1)
                log_denominator = logsumexp(probs_column)

                probs_column -= log_denominator

                likelihood += probs_column[doc_label_index]

                # probs_column is a column vector of the size of the label set, where each of the values are
                probs_column = np.exp(probs_column)

                expected_matrix[:, sparse_feature_vector] += probs_column.reshape(len(self.label_set),1)

            # create a gradient matrix with the pointwise difference between the observed and expected matrices
            gradient_matrix = observed_matrix - expected_matrix
            # pointwise add to the weights matrix the gradient matrix reduced by the learning rate value
            self.weights_matrix += gradient_matrix * learning_rate

            # calculates the accuracy after each minibatch
            accuracy_score_after_update = 0
            for doc in dev_instances:
                # classifies always wrt weights_matrix
                predicted_label = self.classify(doc)
                if predicted_label == doc.label:
                    accuracy_score_after_update += 1

            new_accuracy = accuracy_score_after_update / len(dev_instances)
            print('acc dev', new_accuracy, 'train likelihood', likelihood)


            # if the new accuracy is less than the old accuracy move down the count of chances left before stoppage
            if new_accuracy <= old_accuracy:
                chances_left -= 1
            # if we used up the chances, stop looping
            if chances_left == 0:
                converged = True

            old_accuracy = new_accuracy

    def classify(self, instance):
        """Classifies the given instance based on a given feature set and a previously trained matrix of feature
        weights """

        if "positive" in self.label_set:
            instance_features = instance.features().intersection(self.feature_set)
        else:
            instance_features = instance.features()

        sparse_feature_vector = [self.feature_index_dict[feature] for feature in instance_features]

        probs_column = self.weights_matrix[:, sparse_feature_vector].sum(1)

        argmax_index = np.argmax(probs_column)

        return self.index_label_dict[argmax_index]



    def create_label_set(self, instance):
        """Populates the attributes that contain the set of labels"""
        if 'yelp' in instance.source:
            self.label_set = {'positive', 'negative', 'neutral'}
        elif 'male' in instance.source:
            self.label_set = {'male', 'female'}

        self.index_label_dict = dict(enumerate(self.label_set))
        self.label_index_dict = {value: key for key, value in self.index_label_dict.items()}


    def create_feature_set(self, document_list):
        """Populates attributes of the model with (1) the set of features, (2) a dictionary from indices in the matrices
         to features, and (3) a dict from features to their index in the matrices"""

        if 'yelp' in document_list[0].source:
            # create megadoc of documents
            megadoc = ''
            for doc in document_list:
                megadoc = megadoc + ' ' + doc.data

            megadoc = megadoc.lower()
            tokd_list = word_tokenize(megadoc)
            full_word_dict = collections.OrderedDict(sorted(collections.Counter(tokd_list).items()))
            print('Full vocabulary dict created')

            # create set of words to be used as features (have to appear more than X times in the whole megadoc)
            word_set = set()
            for word in full_word_dict:
                # if the word occurs more than X times, the set size is less than X, the word is not a stopword and it
                # is only composed of alphabetic symbols, add the word to the set
                if full_word_dict[word] > 200 and 8000 > len(word_set) and word not in stopwords.words('english') and word.isalpha():
                    word_set.add(word)

            # save this word_set as an attribute of the MaxEnt object
            self.feature_set = word_set
            print('Word-set for review feature vectors created!')


        elif 'male' in document_list[0].source:
            import string
            self.feature_set = set(string.ascii_letters)
            self.feature_set.add('#')
            print(self.feature_set)

        self.index_feature_dict = dict(enumerate(self.feature_set))
        self.feature_index_dict = {value : key for key,value in self.index_feature_dict.items()}


