import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(
            self, 
            all_word_sequences: dict, 
            all_word_Xlengths: dict, 
            this_word: str,
            n_constant=3,
            min_n_components=2, 
            max_n_components=10,
            random_state=14, 
            verbose=False
        ):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        result = self.n_constant
        lowest_score = np.inf        
        try:
            # folder = KFold(random_state=self.random_state, n_splits=3)
            for s in range(self.min_n_components, self.max_n_components+1):
                model = self.base_model(s)

                logL = model.score(self.X, self.lengths)
                p = s**2 + 2 * s * model.n_features - 1
                logN = math.log(sum(self.lengths))

                bic = -2 * logL + p * logN
                if bic < lowest_score:
                    lowest_score = bic
                    result = s      
        except:
            pass

        return self.base_model(result)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        result = None
        best_score = -np.inf

        try:
            for s in range(self.min_n_components, self.max_n_components+1):
                model = self.base_model(s)
                scores = []
                for word, (X, lenghts) in self.hwords.items():
                    if word != self.this_word:
                        scores.append(model.score(X, lenghts))
                #fixing the formula below
                logL = model.score(self.X, self.lengths)
                dic = logL - (1/(len(self.words)-1)) * sum(scores)
                if dic > best_score:
                    best_score = dic
                    result = model     
        except:
            pass

        return result


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds '''
    def select(self):        
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        result = self.n_constant
        best_score = -np.inf        
        try:
            folder = KFold(random_state=self.random_state, n_splits=3)
            for s in range(self.min_n_components, self.max_n_components+1):
                score_list = []

                for train_indices, test_indices in folder.split(self.sequences):
                    X_train, y_train = combine_sequences(train_indices, self.sequences)
                    X_test, y_test = combine_sequences(test_indices, self.sequences)

                    hmm = GaussianHMM(
                                n_components=s, 
                                n_iter=1000,
                                covariance_type="diag",
                                random_state=self.random_state
                            ).fit(X_train, y_train)

                    score = hmm.score(X_test, y_test)
                    score_list.append(score)

                average_score = np.mean(score_list)
                if average_score > best_score:
                    best_score = average_score
                    result = s       
        except:
            pass

        return self.base_model(result)
