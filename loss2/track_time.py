import sys
import csv
import pandas as pd

# Get the execution time from command-line argument
execution_time = sys.argv[1]

df = pd.DataFrame({'Time': [execution_time]})
df.to_csv('output_data/time.csv', index=False)