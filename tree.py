import numpy as np
import string

from scipy.stats import entropy
from time import time

from scipy import stats

class Tree:
    def __init__( self,
                  params={ 'max_depth' : 30,
                           'min_sample_count' : 10000,
                           'feature_names' : [] } ):
        self.params = params
        self.leaf = None
        self.left = None
        self.right = None
        self.mapping = []
        
    def __len__(self):
        if self.leaf is not None:
            return 1
        else:
            return 1 + len(self.left) + len(self.right)

    def make_leaf(self, Y, parent_proba ):
        if len(Y) < self.params['min_sample_count'] or Y.std() == 0:
            self.leaf = parent_proba
        else:
            self.leaf = np.bincount(Y,minlength=2).astype('float64') / Y.shape[0]
        return

    def make_mapping(self,X,Y):
        for i in range(len(X[0])):
            self.mapping.append({})
        for x,y in zip(X,Y):
            for i in range(len(X[0])):
                if x[i] not in self.mapping[i]:
                     self.mapping[i][x[i]] = { 'visits' : 0, 'clicks' : 0 }
                self.mapping[i][x[i]]['clicks'] += y
                self.mapping[i][x[i]]['visits'] += 1

        all_max = np.zeros(len(X[0]),dtype='float')
        found = False
        for i in range(len(X[0])):
            if len(self.mapping[i]) < 2:
                continue
            found = True
            clicks = np.zeros(len(self.mapping[i]), dtype='float64')
            visits = np.zeros(len(self.mapping[i]), dtype='float64')
            for j,k in enumerate(self.mapping[i]):
                self.mapping[i][k]['click_proba'] = float(self.mapping[i][k]['clicks'])/float(self.mapping[i][k]['visits'])
                visits[j] = self.mapping[i][k]['visits']
                clicks[j] = self.mapping[i][k]['clicks']
                
            all_max[i] = entropy(clicks,visits)

        if not found:
            print "not found"
            return None, None
        
        feature = np.argmax(all_max)
        values = np.zeros( len(self.mapping[feature]), dtype='float64' )
        visits = np.zeros( len(self.mapping[feature]), dtype='float64' )
        for i,k in enumerate(self.mapping[feature]):
             values[i] = self.mapping[feature][k]['click_proba']
             visits[i] = self.mapping[feature][k]['visits']
             
        # are we gaining information
        # if values.std() < 0.001:
        #     print "not enough gain"
        #     return None, None
        #cumsum = np.cumsum(visits[np.argsort(values)])
        #threshold = values[np.argsort(values)[np.argmax(cumsum>=cumsum.max()/2)]]
        print values
        threshold = values.mean()
        #threshold = (values.max() - values.min())/2
        
        return feature, threshold
    
    def fit( self, X, Y, depth=0, parent_proba=None ):
        
        if ( depth == self.params['max_depth']
             or len(X) <= self.params['min_sample_count']
             or Y.std() == 0):
            self.make_leaf( Y, parent_proba )
            return

        proba = np.bincount(Y,minlength=2).astype('float64') / Y.shape[0]
    
        self.feature, self.threshold = self.make_mapping(X,Y)

        if self.feature is None:
            print "no feature found"
            self.make_leaf( Y, parent_proba )
            return
        
        print self.feature, self.params['feature_names'][self.feature], self.threshold
        
        X_left = []
        Y_left = []
        X_right = []
        Y_right = []
        for x,y in zip(X,Y):
            if self.mapping[self.feature][x[self.feature]]['click_proba'] >= self.threshold:
                X_right.append(x)
                Y_right.append(y)
            else:
                X_left.append(x)
                Y_left.append(y)

        print "size of children:", len(X_left), len(X_right)
        # if ( len(X_left) <= self.params['min_sample_count']
        #      or len(X_right) <= self.params['min_sample_count'] ):
        #     self.make_leaf( Y, parent_proba )
        #     return

        Y_left = np.array(Y_left)
        Y_right = np.array(Y_right)
            
        self.left = Tree(self.params)
        self.right = Tree(self.params)

        self.left_weight = float(len(Y_left))/float(len(Y))
        self.right_weight = float(len(Y_right))/float(len(Y))
        
        self.left.fit( X_left,
                       Y_left,
                       depth+1,
                       proba )
        self.right.fit( X_right,
                        Y_right,
                        depth+1,
                        proba )

        return

    def _predict(self,x):
        if self.leaf is not None:
            return self.leaf
        else:
            if x[self.feature] in self.mapping[self.feature]:
                if self.mapping[self.feature][x[self.feature]]['click_proba'] >= self.threshold:
                    return self.right._predict(x)
                else:
                    return self.left._predict(x)
            else:
                #return 0.5*( self.left._predict(x) + self.right._predict(x) )
                return self.left_weight*self.left._predict(x) + self.right_weight*self.right._predict(x)
            
    def predict_proba(self,X):
        Y = np.zeros( (len(X),2), dtype='float64' )
        for i in xrange(len(X)):
            Y[i] = self._predict(X[i])
        return Y

