import warnings
from asl_data import SinglesData
import numpy as np

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    
    xs = test_set.get_all_Xlengths()        
    for xleng in xs.values():
        highest_prob = -np.inf
        best_word = ""
        probs = {} # dict to append to the result

        for word, model in models.items():
            probs[word] = -np.inf #initializes as minus infinity
            try:
                prob = model.score(xleng[0], xleng[1])
                probs[word] = prob
                if prob > highest_prob:
                    highest_prob = prob
                    best_word = word
            except:
                pass  
        
        probabilities.append(probs)
        guesses.append(best_word)

    return probabilities, guesses
    
