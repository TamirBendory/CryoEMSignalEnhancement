#!/bin/bash
# TO RUN THIS BASH FILE: . '/scratch/guysharon/Work/Python/cryo_class_average/src/Test_cryo_class_average.sh'

# install package
cd /scratch/guysharon/Work/Python/cryo_class_average
python setup.py sdist 
pip install dist/cryo_class_average-1.0.tar.gz

# run cryo_class_average
starfile='/scratch/guysharon/Work/starfiles/dataset_10028'
output='/scratch/guysharon/Work/Python/saved_test_data/ca_10028_from_user'
'/home/guysharon/.local/bin/cryo_class_average' --starfile $starfile --output $output
