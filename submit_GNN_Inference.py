import os
import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', help='Directory that includes the data to be processed [NANOAOD]')
parser.add_argument('--output_dir', help='Directory of output')
parser.add_argument('--model', help='path of the model')
args = parser.parse_args()

f = open("submitGNN.sh",'w')

for s in os.listdir(args.input_dir):
  s_path = os.path.join(args.input_dir,s)
  fileNames = []
  for root, dirs, files in os.walk(s_path):
    for ifile in files:
      if (ifile.endswith('.pt')):
        fileNames.append(os.path.join(root,ifile)) 
  for fn in fileNames:
    cmd = "python Inference_GNN_per_file.py --input_file {} --output_dir {} --model {};".format(fn,os.path.join(args.output_dir,s),args.model)
    f.write(cmd+'\n')

  #cmd = "python Inference_GNN_per_sample.py --input_dir {} --output_dir {} --model {};".format(os.path.join(args.input_dir,s),os.path.join(args.output_dir,s),args.model)
  #f.write(cmd+'\n')

f.close()
