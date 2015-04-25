#!/bin/bash

set -x
set -e

function mysbatch {
    sbatch --mem=16G -c 2 -p long --exclude biomedia10,predict[1-3,6,8] --wrap="$*"
}

mkdir -p output/tree

for name in remaining_websites remaining_apps # main_website main_app
do
    python train_tree.py \
        --input data/train_${name}.gz \
        --run data/test_${name}.gz \
        --output output/tree/${name} &

done
