#!/bin/bash

env_name="tf_test"

source activate $env_name
iteration=$1
output_folder="output_data_${iteration}"
# Start tracking time
start_time=$(date +%s)

# specify variable for kernel_type, model_type, training_type and system and then use them as arguments for the python scripts
model_type="ensemble"
training_type="continuous"
system="lorenz"
# Execute the init Python script
python init.py --system $system --output_folder $output_folder

# Loop x times

for ((i=1; i<=2; i++))
do
    # Execute the exec Python script
    python3 active_learning.py --system $system --output_folder $output_folder

    if [ $? -ne 0 ]; then
      break
    fi
done

# Calculate the execution time
end_time=$(date +%s)
execution_time=$((end_time - start_time))

# Pass the execution time to the Python script
python3 utils/track_time.py --time $execution_time --output_folder $output_folder
#python3 utils/track_time.py "$execution_time"
