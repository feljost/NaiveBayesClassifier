import numpy as np
from scipy.stats import norm, bernoulli

# Distribution for continuous features
class ContFeatureParam:
    
    # Estimate the mean and the standard deviation
    def estimate(self, X): 
        X = X[~np.isnan(X)]
        self.estimate_loc, self.estimate_scale = norm.fit(X)

    # Get probability of observing the value conditioned on mean and standard deviation
    def get_probability(self, val):
        prob = np.log(norm.pdf(val, self.estimate_loc, self.estimate_scale+1e-6)+1e-6)
        return prob


# Distribution for binary features
class BinFeatureParam:
    # Estimatethe probability of the value being 1 (using the percentage of ones in all of x)
    def estimate(self, X):
        X = X[~np.isnan(X)]
        num_ones = np.sum(X)
        self.p = num_ones / X.size

    # Get probability of observing the value conditioned on probabilities defined above
    def get_probability(self, val):
        if val == 0:
            return np.log(1-self.p+1e-6)
        else:
            return np.log(self.p+1e-6)

# Distribution for categorical features
class CatFeatureParam:
    # Estimate the probability of the values using the percentage of the values in all of x
    # This is works the same as for the binary features, but looping through the different classes
    def estimate(self, X):
        X = X[~np.isnan(X)]
        (unique, counts) = np.unique(X, return_counts=True)
        total = X.size
        self.p = {}
        for u, c in zip(unique, counts):
            self.p[u] = c / total

    # Get probability of observing the value conditioned on probabilities defined above
    def get_probability(self, val):
        if val in self.p:
            return np.log(self.p[val]+1e-6)
        else: #only happens if training data is too small
            return np.log(0+1e-6)

class NBC:
    # Inputs:
    # feature_types: the array of the types of the features, e.g., feature_types=['r', 'r', 'r', 'r']
    # num_classes: number of classes of labels
    def __init__(self, feature_types=[], num_classes=13):
        self.feature_types=feature_types
        self.num_classes=num_classes

    
    # The function uses the input data to estimate all the parameters of the NBC
    def fit(self, X, y):

        self.p_label = self.calculateLabelProbability(y)
        
        self.model = {}
        for label in range(self.num_classes):
            X_label = X[y == label]
            self.model[label] = self.estimateFeatureParameters(X_label)
            
        # No need for the denominator, since we just want to compare them between different labels.
        # The denominator is a constant across all labels.
                
    # The function takes the data X as input, and predicts the class for the data
    def predict(self, X):
        y_predict = []
        for x in X:
            probabilities = {}
            for label in self.p_label:
                prior = self.p_label[label]
                fparams = self.model[label]
                likelihood = 0
                for fparam, feat in zip(fparams, x):
                    likelihood += fparam.get_probability(feat)
                probabilities[label] = prior + likelihood #numerator (+ because we're in log space)
            y_predict.append(max(probabilities, key=probabilities.get)) # choose the class with the highest probability
        return y_predict
        
    def calculateLabelProbability(self, y):
        # can't handle NaN but label set should not containt any NaN.
        # Alternatively, NaN could just be seen as another category.
        (unique, counts) = np.unique(y, return_counts=True)
        total = y.size
        p_y = {}
        for u, c in zip(unique, counts):
            p_y[u] = np.log(c / total)
        return p_y
    
    def estimateFeatureParameters(self, X):
        model = []
        for ftype, param in zip(self.feature_types, X.transpose()):
            if ftype == 'b':
                fparam = BinFeatureParam()
            elif ftype == 'r':
                fparam = ContFeatureParam()
            elif ftype == 'c':
                fparam = CatFeatureParam()
            else:
                fparam = None
                print('no such feature type ' + str(ftype))
            if fparam:
                fparam.estimate(param)
                model.append(fparam)
        return model