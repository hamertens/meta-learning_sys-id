# Define specifications for the model and dynamic system
FOLDER_FILEPATH = "/home/hansm/active_learning/ICS_635_project/Ensemble_cont_acc_acquisition/"
DATA_FILEPATH = "/home/hansm/active_learning/ICS_635_project/data/"
INPUT_DIMENSIONALITY = 3
OUTPUT_DIMENSIONALITY = 3
DATAFRAME_COLUMNS  = ["x", "y", "z"]

#name of anaconda environment
env_name = "meta_learning" 

ensemble_bash_filepath = "/home/hansm/active_learning/ICS_635_project/Ensemble_cont_acc_acquisition/ensemble.sh"

# Path to the init Python script
python_script_init="/home/hansm/active_learning/ICS_635_project/Ensemble_cont_acc_acquisition/init.py"

# Path to the loop Python script
python_script_exec="/home/hansm/active_learning/ICS_635_project/Ensemble_cont_acc_acquisition/exec.py"

# Path to the Bayesian optimization python script
python_script_bayes="/home/hansm/active_learning/ICS_635_project/Ensemble_cont_acc_acquisition/scikit_opt.py"

# List of file paths
filepaths = [ensemble_bash_filepath, python_script_init, python_script_exec, env_name, python_script_bayes]

if __name__ == "__main__":
    # Write the file paths to a temporary file, each on a new line
    with open("filepaths.txt", "w") as f:
        for path in filepaths:
            f.write(path + "\n")