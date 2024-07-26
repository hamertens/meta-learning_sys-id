import sys
import csv
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Provide output folder and execution time")
parser.add_argument("-t", "--time", type=str, required=True, help="Time (required)")
parser.add_argument("-of", "--output_folder", type=str, required=True, help="Output Folder (required)")

args = parser.parse_args()
execution_time = args.time
output_folder = args.output_folder

# Get the execution time from command-line argument
#execution_time = sys.argv[1]

df = pd.DataFrame({'Time': [execution_time]})
df.to_csv(output_folder + '/time.csv', index=False)