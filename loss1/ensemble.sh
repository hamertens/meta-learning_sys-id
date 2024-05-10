#!/bin/bash


source activate tf_gpu

# Start tracking time
start_time=$(date +%s)

# Read the file paths from the temporary file into an array
mapfile -t filepaths < filepaths.txt

# Execute the init Python script
python ${filepaths[1]}


# Loop x times

for ((i=1; i<=400; i++))
do
    # Execute the exec Python script
    python3 ${filepaths[2]}

    if [ $? -ne 0 ]; then
      break
    fi
done

# Calculate the execution time
end_time=$(date +%s)
execution_time=$((end_time - start_time))

# Pass the execution time to the Python script
python3 track_time.py "$execution_time"

source deactivate
