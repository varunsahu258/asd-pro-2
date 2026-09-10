# HCAN adaptation note

This project implements the HCAN architecture of Shao, Fu & Chen (2023),
*A heterogeneous graph convolutional attention network method for classification
of autism spectrum disorder*, BMC Bioinformatics 24, 98,
https://doi.org/10.1186/s12859-023-05241-2.  It removes the original **site**
meta-path and retains only sex and handedness edges. Site edges would expose
held-out site identity during LOSO evaluation and cannot be constructed without
leakage for a truly external subject.

The original paper reports 82.9% accuracy using 10-fold cross-validation that
mixes all sites, with no external validation. Those results are not directly
comparable with this project's LOSO or external-site estimates: the evaluation
protocols answer different generalization questions.
