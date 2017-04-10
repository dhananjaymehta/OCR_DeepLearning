__author__ = 'Dhananjay Mehta/Swapnil Kumar'
import cPickle
import theano
from matplotlib import pyplot as plt


class PredictCharacter():

    def predict_character(self, candidates, best_model = 'best_model_char_num.pkl'):
        classifier = cPickle.load(open(best_model))
        predict_model = theano.function(inputs=[classifier.input],outputs=classifier.y_pred)
        for single_x in candidates:
            d = single_x.reshape(28,28)
            print ("Classified as: %s" % chr(predict_model([single_x])))
            plt.imshow(d, interpolation="nearest")
            plt.show()

