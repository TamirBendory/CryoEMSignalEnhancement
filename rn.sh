#!/bin/sh
# /usr/local.cc/bin/matlabr2018b -c 27000@lm8-2 &
# . '/scratch/guysharon/Work/CryoEMSignalEnhancement/rn.sh'

# reinstall package
cd /scratch/guysharon/Work/CryoEMSignalEnhancement
python setup.py sdist 
pip install dist/cryo_signal_enhance-1.0.tar.gz
clear

'/home/guysharon/.local/bin/cryo_signal_enhance' --starfile /scratch/guysharon/Work/starfiles/dataset_10028 --output /scratch/guysharon/Work/Python/saved_test_data/tmp --basis_file_in /scratch/guysharon/Work/Python/saved_test_data/basis_10028 --sample True
done

# test starfiles that I created
#return
main='/home/guysharon/.local/bin/cryo_class_average'
datasets=("DunaLarge" "10028" "10073" "10081" "10061" "10049" "10123" "10272")
for dataset in ${datasets[@]}; do
	query="--starfile /scratch/guysharon/Work/starfiles/dataset_$dataset --output /scratch/guysharon/Work/Python/saved_test_data/ca_$dataset --basis_file_out /scratch/guysharon/Work/Python/saved_test_data/basis_$dataset"
	eval $main $query
done

# test starfiles that I did not create
'/home/guysharon/.local/bin/cryo_class_average' --starfile /data/yoelsh/datasets/10028/data/Particles/shiny_2sets.star --output /scratch/guysharon/Work/Python/saved_test_data/ca_10028_from_yoel_starfile --debug_verbose true

# test random starfiles
starfiles=("10032/data/particles_phaseflipped/ynai_particles.star" "10063/particles/ring11.star" "10065/all_images.star" "10107/data/shiny_new.star" "10124/LMNG_final_fixed.star" "10421/03_Final_Particle_Stack/BIU_HuApo_Final.star" "dls0519/relion/CtfRefine/job017/particles_ctf_refine.star" "embl0919/embl0919_from_eman_class2/Extract/job004/particles.star" "embl1119/relion_after_eman2/CtfRefine/job013/particles_ctf_refine.star" "embl1219/relion/CtfRefine/job055/particles_ctf_refine.star" "esrf0919/Extract/job011/particles.star" "esrf1118/relion_from_eman/CtfRefine/job012/particles_ctf_refine.star" "esrf1118_clhetero/Extract/job220/particles.star" "kVstack/select008_fixed.star" "ranzGroEL/particles.star")

i=1
for starfile in ${starfiles[@]}; do
	query="--starfile /data/yoelsh/datasets/$starfile --output /scratch/guysharon/Work/Python/saved_test_data/randoms/out$i --debug_verbose true"
	eval $main $query
	((i=i+1))
done

