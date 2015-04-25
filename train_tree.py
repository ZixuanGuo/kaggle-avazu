#!/usr/bin/python

# time feature: weekday+hour/24 ?

# calibration: is there a bias in our predictions?

# find best random seed

from __future__ import division

import numpy as np
import gzip
from datetime import datetime
from time import time
import cPickle as pickle

from sklearn import ensemble
from sklearn.metrics import log_loss
from sklearn.base import BaseEstimator

from tree import Tree

import joblib

import argparse

np.random.seed(9467135)

parser = argparse.ArgumentParser(
        description='' )
parser.add_argument( "--input", type=str )
parser.add_argument( "--run", type=str )
parser.add_argument( "--output", type=str )
parser.add_argument( "--verbose", type=int, default=0 )
parser.add_argument( '--debug', action="store_true", default=False )
args = parser.parse_args()

def read_time(t):
    return datetime( int('20'+t[:2]),
                     int(t[2:4]),
                     int(t[4:6]),
                     int(t[6:]) )

if args.run is not None and args.output is None:
    print "need to give --output"
    exit(1)

f = gzip.open(args.input, 'rb')

# begin: 14102100; end: 14103023 (YYMMDDHH)
if args.debug:
    testing_time = 14102101
    if args.run is None:
        stop_testing = 14102102
    else:
        stop_testing = 14102101
elif args.run is None:
    testing_time = 14102800
    stop_testing = None #14102102
else:
    testing_time = None
    stop_testing = None

time_shift = pickle.load(open("mapping/time_shift.pk","rb"))

header = None
feature_names = []
X_train = []
Y_train = []
X_test = []
Y_test = []

for line in f:
    line = line.rstrip().split(',')

    # read header
    if header is None:
        header = line
        feature_names = line[2:]
        feature_names.append("day")
        continue
    
    # what time is it?
    t = int(line[2])

    # day
    d = read_time(line[2]).weekday()

    # hour
    line[2] = line[2][-2:]

    k = ""
    for _c in [5,6,7,8,9,10]:
        k += line[_c] + "_"
        
    line[2] = str( ( (int(line[2])-time_shift[k]) % 24 ) )
        
    line[11] = ''
    line[12] = ''

    #line.append(d)

    if testing_time is None or t < testing_time:
        X_train.append(line[2:])
        Y_train.append(int(line[1]))
    else:
        X_test.append(line[2:])
        Y_test.append(int(line[1]))
    if stop_testing is not None and t >= stop_testing:
        break

f.close()

Y_train = np.array( Y_train, dtype='int32' )
Y_test = np.array( Y_test, dtype='int32' )

print "starting training"

classifier = Tree(params={ 'max_depth' : 20,
                           'min_sample_count' : 10000,
                           'feature_names' : feature_names } )
classifier.fit( X_train, Y_train )

print "starting testing"

if args.run is None:
    Y_pred = classifier.predict_proba( X_test )

    print "log loss:", log_loss( Y_test, Y_pred, normalize=True )
    
    exit(0)

f = gzip.open(args.run, 'rb')
header = None
X = []
ad_ids = []
for line in f:
    line = line.rstrip().split(',')

    # read header
    if header is None:
        header = line
        continue

    # save ad_id
    ad_ids.append( line[0] )
    
    # what time is it?
    t = int(line[1])

    # day
    d = read_time(line[1]).weekday()

    # hour
    line[1] = line[1][-2:]
    line[10] = ''
    line[11] = ''

    k = ""
    for _c in [5,6,7,8,9,10]:
        k += line[_c-1] + "_"
        
    line[1] = str( ( (int(line[1])-time_shift[k]) % 24 ) )

    #line.append(d)
    
    X.append(line[1:])

f.close()

Y_pred = classifier.predict_proba(X)

pickle.dump( ad_ids, open( args.output + "_ids.pk", "wb" ) )
np.save( args.output + "_pred.npy", Y_pred )
