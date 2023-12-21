# How to reproduce our results

## 1. Create a Virtual Environment 
### Option 1: Conda virtual environment built using the [environment.yml](environment.yml) file.
To create a conda virtual environment with the yml file, run the following command in the terminal 
```
conda env create -f environment.yml
```
### Option 2: Create a virtual environment and install the required packages using the [requirements.txt](requirements.txt) file. 
Create a python virtual environment (we use Python 3.9.18), then install the requirements by running the following command 
```
pip3 install -r requirements. txt
```

## 2. Generate the Synthetic Data
As the size of the synthetic data is too large to be added to the GitHub, it has to be generated using the [create_synthetic_data.py](create_synthetic_data.py) script. The generated data is then stored in the [synthetic_data](synthetic_data) directory. 

## 3. (Optional) Train the MLP model
This can be done by running the [train_mlp.py](train_mlp.py) script. You have to provide the directory in which the input data is stored (the matrices F, H and S) as well as the model's name as follows: 
```
python3 train_mlp.py <path_to_input_data> <model_name>
```
Example usage: 
```
python3 train_mlp.py synthetic_data my_model
```


## 4. Test the MLP model
To do this, simply run the [model_test.ipynb](model_test.ipynb) notebook. Be sure to modify the weights and layer configurations accordingly if you tried training the model with different parameters than the default ones provided in the python script. Make sure to use a Jupyter Kernel that has all the requirements specified to build the virtual environment. 
To obtain an excel file with the metrics used to assess the quality of the model, you can run the [compute_local_metrics.py](compute_local_metrics.py) script. 