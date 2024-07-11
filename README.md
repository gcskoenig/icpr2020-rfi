# Relative Feature Importance
by Gunnar König, Christoph Molnar, Bernd Bischl, Moritz Grosse-Wentrup [[arXiv]](https://arxiv.org/abs/2007.08283)  

## Contents

This repository contains the code accompagnying the paper "Relative Feature Importance", submitted at ICPR2020.  

- datagen.py: code for the sampling of the dataset used in the examples
- rfi.py: functions for the computation of relative feature importance
- ex1_confounding: code for the example with variables outside training feature set
- ex2_chain: code for indirect influence example

## Dependencies

matplotlib==3.2.2  
seaborn==0.10.1  
torch==1.5.0  
pyro_ppl==1.3.1  
pandas==1.0.5  
numpy==1.18.1  
scipy==1.4.1  
pyro==3.16  
scikit_learn==0.23.1  

## Implementation of RFI

An updated implementation of RFI can be found in the `fippy` package on [https://github.com/gcskoenig/fippy](https://github.com/gcskoenig/fippy).
