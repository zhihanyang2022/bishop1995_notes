# My notes on Bishop 1995

Remarks:

- This repo contains my notes and demos for understanding Neural Networks for Pattern Recognition (Bishop, 1995). 
- This repo is not meant for sharing, but resources here are well-documented and using them is more than welcomed. 

Where things are located:

- `painted_data_to_numpy_array.md` is stored in `./c3_single_layer_network/perceptron`. This document shows how to convert a CSV (containing data painted and formatted in Orange https://orange.biolab.si/) into Numpy arrays. If you have not used Orange before, it contains an excellent tool called "Paint Data" for creating custom 2D data sets for testing models. Of course, it has other functionalities that I seldom use. Examples of CSVs painted in Orange:
    - `./c3_single_layer_network/least_square_method/regression_data.csv`
    - `./c3_single_layer_network/perceptron/data_not_linearly_separable.csv`
    - `./c4_multi_layer_perceptron/circuluar_data.csv`
    - `./c4_multi_layer_perceptron/data_sophisticated_decision_boundary.csv`

- To learn about how to use modules (with callback and lr-finding functionalities) in `./modules_for_nn_training`, see `./c4_multi_layer_perceptron/two_layer_network_universality_and_jacobian_demo.ipynb`.
