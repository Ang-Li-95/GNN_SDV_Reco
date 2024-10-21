import os
import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', help='Directory that includes the data to be processed [NANOAOD]')
parser.add_argument('--output_dir', help='Directory of output')
args = parser.parse_args()

f = open("submitscript.sh",'w')

for s in os.listdir(args.input_dir):
  s_path = os.path.join(args.input_dir,s)
  cmd = "python makeplots.py --input_dir {} --output_dir {};".format(s_path,os.path.join(args.output_dir,s))
  #cmd = "python makeclus.py --input_dir {} --output_dir {};".format(s_path,os.path.join(args.output_dir,s))
  f.write(cmd+'\n')
  

  #cmd = "python Inference_EMB_per_sample.py --input_dir {} --output_dir {} --model {};".format(os.path.join(args.input_dir,s),os.path.join(args.output_dir,s),args.model)
  #f.write(cmd+'\n')

f.close()
