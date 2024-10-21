import os
import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', help='Directory of output')
args = parser.parse_args()

json_path = '/users/ang.li/public/SoftDV/CMSSW_13_3_0/src/SoftDisplacedVertices/Samples/json/CustomNanoAOD_v3_bkg.json'

with open(json_path,'r') as jf:
  d = json.load(jf)["CustomNanoAOD"]["dir"]

f = open("submit.sh",'w')
for s in d:
  fileNames = []
  for root, dirs, files in os.walk(d[s]):
    for ifile in files:
      if (ifile.endswith('.root')):
        fileNames.append(os.path.join(root,ifile)) 
  for fn in fileNames:
    cmd = "python make_data_3.py --input_file {} --output_dir {};".format(fn,os.path.join(args.output_dir,s))
    f.write(cmd+'\n')
