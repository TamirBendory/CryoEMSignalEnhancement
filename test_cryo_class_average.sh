#!/bin/sh
# /usr/local.cc/bin/matlabr2018b -c 27000@lm8-2 &
# . '/scratch/guysharon/Work/Python/cryo_class_average/test_cryo_class_average.sh'

# reinstall package
cd /scratch/guysharon/Work/Python/cryo_class_average
python setup.py sdist 
pip install dist/cryo_class_average-1.0.tar.gz
clear

'/home/guysharon/.local/bin/cryo_class_average' --starfile /scratch/guysharon/Work/starfiles/dataset_10028 --output tmp --basis_file_in /scratch/guysharon/Work/Python/saved_test_data/basis_10028.pkl --N 10000 --num_class_avg 100 --em_num_input 200 --num_classes 200

