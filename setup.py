from setuptools import setup, find_packages

setup(
    name='cryo_signal_enhance',
    version='1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'cryo_signal_enhance = src.__main__:main',
        ]
    },
    install_requires=[
        'numpy',
        'cupy',
        'scipy',
        'mrcfile',
        'psutil',
        'pickle5',
        'progress',
        ],
)