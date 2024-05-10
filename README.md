# meta-learning_sys-id

This is the implementation for the ICS 635 final project using meta learning for system identification with minimum samples. For both loss functions a folder is created where the respective optimization python script can be executed to reproduce the results. The optimization script executes a bash script that runs the active learning loop. This is done to avoid memory allocation in bash script. The network architecture can be changed by adapting the model_function file.

To run the active learning the tf_gpu environment is recommended and for the bayesian optimization the mtea_learning environment. Both environments can be created from the yml files provided in this repository.
