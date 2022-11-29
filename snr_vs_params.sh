#!/bin/sh
# /usr/local.cc/bin/matlabr2018b -c 27000@lm8-2 &
# . '/scratch/guysharon/Work/Python/cryo_class_average/snr_vs_params.sh'

# reinstall package
cd /scratch/guysharon/Work/Python/cryo_class_average
python setup.py sdist 
pip install dist/cryo_class_average-1.0.tar.gz
clear

# init params
datasets=("10028" "10073" "10081" "10061")
datasets=("10061")
n_coeffs=(200 300 500 1000)
n_coeffs=(300)
n_classes=(1500 3000 5000)
n_classes=(5000)
class_sizes=(300 500 700)
class_sizes=(700)
main='/home/guysharon/.local/bin/cryo_class_average'

# loop 1
for dataset in ${datasets[@]}; do
for n_coeff in ${n_coeffs[@]}; do
	calc_new_basis=true
for n_class in ${n_classes[@]}; do
for class_size in ${class_sizes[@]}; do
	if [ "$calc_new_basis" = true ] ; then
	    	query="--starfile /scratch/guysharon/Work/starfiles/dataset_$dataset --output /scratch/guysharon/Work/Python/saved_test_data/statistics/ca_${dataset}_${n_coeff}_${n_class}_${class_size} --num_coeffs $n_coeff --num_classes $n_class --class_size $class_size --basis_file_out /scratch/guysharon/Work/Python/saved_test_data/statistics/tmp_basis"
	else
		query="--starfile /scratch/guysharon/Work/starfiles/dataset_$dataset --output /scratch/guysharon/Work/Python/saved_test_data/statistics/ca_${dataset}_${n_coeff}_${n_class}_${class_size} --num_coeffs $n_coeff --num_classes $n_class --class_size $class_size --basis_file_in /scratch/guysharon/Work/Python/saved_test_data/statistics/tmp_basis"
	fi
	eval $main $query
	calc_new_basis=false
done
done
done
done

# loop 2
num_inputs=(20 50 100 150 200 300 400 500)
for dataset in ${datasets[@]}; do
	calc_new_basis=true
for n_inputs in ${num_inputs[@]}; do
	if [ "$calc_new_basis" = true ] ; then
	    	query="--starfile /scratch/guysharon/Work/starfiles/dataset_$dataset --output /scratch/guysharon/Work/Python/saved_test_data/statistics/ca_${dataset}_${n_inputs} --num_classes 5000 --class_size 500 --em_num_inputs ${n_inputs} --basis_file_out /scratch/guysharon/Work/Python/saved_test_data/statistics/tmp_basis"
	else
	    	query="--starfile /scratch/guysharon/Work/starfiles/dataset_$dataset --output /scratch/guysharon/Work/Python/saved_test_data/statistics/ca_${dataset}_${n_inputs} --num_classes 5000 --class_size 500 --em_num_inputs ${n_inputs} --basis_file_in /scratch/guysharon/Work/Python/saved_test_data/statistics/tmp_basis"
	fi
	#eval $main $query
	calc_new_basis=false
done
done
