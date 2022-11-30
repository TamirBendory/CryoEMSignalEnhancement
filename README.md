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
--starfile starfile_full_path
```


