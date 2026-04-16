# Logistic-Stable-Distillation
Work in progress code and datasets for my masters dissertation 

The method is derived from David Steinsaltz's stable distillation (SD) algorithm: a newly proposed statistical test for global null hypothesis testing in high-dimensional sparse regimes. The code focuses on logistic regression models. The data is from The Cancer Genome Atlas, utilising a corpus of 4,066 bowel cancer patients' gene expression counts.

The latest version uses the binary_x_v2.py to run single logistic regression SD and binary_multinomial.py (currently WIP) to run multinomial logistic regression SD. U_val_verifier.py is dependent on binary_x_v2.py to run an aggregation of the SD process and merge emitted p-values to a single p-value. Soon, a multinomial version will be added.
