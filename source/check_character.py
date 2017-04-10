import cPickle
import theano
from matplotlib import pyplot as plt


class CheckCharacter:

    def check_character(self, candidates, best_model = 'best_model_char_num_check.pkl'):
        valid_candidates = list()
        classifier = cPickle.load(open(best_model))

        predict_model = theano.function(inputs=[classifier.input],outputs=classifier.y_pred)

        for single_x in candidates:
            d = single_x.reshape(28,28)
            val = predict_model([single_x])
            print ("Classified as: %s" % val)
            plt.imshow(d, interpolation="nearest")
            if val == 1:
                valid_candidates.append(single_x)
            plt.show()

        return valid_candidates

