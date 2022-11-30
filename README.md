# CryoEMSignalEnhancement

## About
This project enhances the SNR of Cryo-EM images, using 2-D
classification and expectation-maximization.

## Installation
```
pip install cryo_class_average
```

## Arguments
The package recieves a mandatory starfile (.star) as its input.
The starfile should be pointing to images of particle (post particle-picking)
under the field ```_rlnImageName```.

For CTF correction, the starfile should also include the following parameters:
```
- _rlnAmplitudeContrast
- _rlnSphericalAberration
- _rlnDefocusAngle
- _rlnDefocusV
- _rlnDefocusU
- _rlnVoltage
```

The starfile is given using the following
```
--starfile starfile_path
```

The output is an mrcs (.mrcs) file containing the images with
enhanced SNR, and is given by
```
--output output_path
```

Additional optional arguments are:
```
--basis_file_out     Write sPCA basis to file, allowing to skip preprocessing next time (.pkl)
--basis_file_in      Optional input file for sPCA basis. When given, skips preprocessing (.pkl)
--N                  Number of images to take from starfile
--verbose            Print progress to screen
--downsample         Preprocessed image size
--batch_size         Batch size for PSD noise estimation
--num_coeffs         Number of sPCA coefficient
--class_theta_step   Sampling rate for the in-plane angle
--num_classes        Number of classes to find (<= N)
--class_size         Number of images in each class (<= N)
--class_gpu          Use available gpu for matrix calculations (recommended)
                            -1       = don't use gpu
                             0       = select gpu with most available memory
                             1,2,... = use specific gpu
--ctf               Perform CTF correction
--iter              Maximum number of iterations to perform
--norm              Perform normalisation-error correction
--em_num_inputs     Number of images to use for EM (taken from best of each class) (<= class_size)
--em_par            Perform EM parallelly (recommended)
--num_class_avg     Number of class averages to produce (<= num_classes)
```





