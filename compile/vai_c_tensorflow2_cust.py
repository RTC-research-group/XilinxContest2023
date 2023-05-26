#!/usr/bin/env python
# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import json
import argparse


def print_banner(str):
  print("**************************************************")
  print("* {}".format(str))
  print("**************************************************")


def printWarning(str):
  print("WARNING: {}".format(str))


def printError(str):
  print("ERROR: {}".format(str))


parameters = [
    ("-m", "--model", str, None, 'store', "h5 model file"),
    ("-a", "--arch", str, None, 'store', "json file"),
    ("-o", "--output_dir", str, "./", 'store', "output directory"),
    ("-n", "--net_name", str, "deploy", 'store', "prefix-name for the outputs"),
    ("-e", "--options", str, "", 'store',
     "extra options. Use --options '{\"input_shape\": \"1,224,224,3\"}' to specify input shape manually, or if there are more than 1 inputs, use --options '{\"input_shape\": {\"data_op0\": \"1,224,224,3\", \"data_op1\": \"1,112,112,3\"}}'. Use --options '{\"batchsize\": 4}' to modify the batchsize. Use --options '{\"plugins\": \"plugin0,plugin1\"}' to specify plugin libraries. Use --options '{\"output_ops\": \"op_name0,op_name1\"}' to specify output ops. Use --options '{\"hd_opt\": \"true\"}' to enable special optimization for HD input, the default is false."
    ),
]


def default_compiler_arg_parser(params=parameters):
  # NOTE: Not all arguments are passed as separate variables to backend functions.
  # Removing a line here may cause errors until we can completely remove the args parameter from backend functions
  parser = argparse.ArgumentParser()
  for x in params:
    if x[2] is bool:
      if x[0] is not None:
        parser.add_argument(x[0], x[1], default=x[3], action=x[4], help=x[5])
      else:
        parser.add_argument(x[1], default=x[3], action=x[4], help=x[5])
    elif x[0] is not None:
      parser.add_argument(
          x[0], x[1], type=x[2], default=x[3], action=x[4], help=x[5])
    else:
      parser.add_argument(x[1], type=x[2], default=x[3], action=x[4], help=x[5])
  return parser


class VAI_TensorFlow_Frontend():

  def __init__(self, args=None):
    self.c = None
    self.res = None
    self.args = args
    self.frontend = ''

    if args is None:
      ## load default arguments
      parser = default_compiler_arg_parser()
      args = parser.parse_args([])  # Parse an empty list

    data = None

    try:
      with open(args.arch) as json_data:
        if json_data:
          data = json.load(json_data)
    except:
      printError("Unable to open file {}".format(args.arch))

    param = {
        'parser': 'tensorflow2',
        'h5_file': args.model,
        'output_dir': args.output_dir,
        'net_name': args.net_name
    }
    if args.options != "":
      extra_options = eval(args.options)
      param.update(extra_options)

    if data and 'target' in data and data['target'][0:9] in [
        'DPUCAHX8H', 'DPUCZDX8G', 'DPUCAHX8L', 'DPUCVDX8G', 'DPUCVDX8H',
        'DPUCADF8H'
    ]:
      param['target'] = data['target']
    elif data and 'target' in data and data['target'][0:10] in [
        'DPUCV2DX8G'
    ]:
      param['target'] = data['target']
    elif data and 'fingerprint' in data:
      param['fingerprint'] = data['fingerprint']

    if 'target' in param or 'fingerprint' in param:
      
      import sys
      sys.path.append('/opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.7/site-packages/vaic')
      from xcompiler_interface import xcompiler
      
      self.c = xcompiler(param)
      self.frontend = 'xcompiler'
    else:
      self.c = None

  def compile(self):
    if self.c is None:
      printError("NO FRONT END SPECIFIED")
      return None
    return self.c.compile()


if __name__ == "__main__":
#   import debugpy

#   debugpy.listen(("0.0.0.0", 5678))
#   print("Waiting for client to attach...")
#   debugpy.wait_for_client()
  print_banner('VITIS_AI Compilation - Xilinx Inc.')
  parser = default_compiler_arg_parser()
  args = parser.parse_args()
  if not args.model is None and not args.arch is None:
    compiler = VAI_TensorFlow_Frontend(args)
    compiler.compile()
  else:
    print("vai_c_tensorflow2.py -h # for use")
