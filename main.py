import subprocess 
import sys

args = sys.argv

arguments = args[1:]

subprocess.run(['python3', 'train.py'])

if len(arguments) >= 1:
    subprocess.run(['python3', 'extract.py', arguments[0]])
else:
    subprocess.run(['python3', 'extract.py'])

subprocess.run(['python3', 'prediction.py'])