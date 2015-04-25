#!/usr/bin/python

import sys
from glob import glob
import gzip
import numpy as np
import cPickle as pickle

output = sys.argv[1]

filenames = zip( [ "main_app_ids.pk",
                   "remaining_apps_ids.pk",
                   "main_website_ids.pk",
                   "remaining_websites_ids.pk" ],
                 [ "main_app_pred.npy",
                   "remaining_apps_pred.npy",
                   "main_website_pred.npy",
                   "remaining_websites_pred.npy" ] )

tree = {}

for f_ids, f_pred in filenames:
    Y_pred = np.load( "output/tree/"+f_pred )[:,1]
    ad_ids = pickle.load( open("output/tree/"+f_ids,"rb") )
    for i,y in zip(ad_ids, Y_pred):
        tree[i] = y

f = gzip.open( output, "wb" )
f.write( "id,click\n" )
for i in sorted(tree.keys()):
    f.write(i+","+str(tree[i])+"\n")
f.close()
