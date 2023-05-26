import json
import os
import re
import sys
import argparse
import time
import pdb
import random
from pytorch_nndct.apis import torch_quantizer
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18
from tqdm import tqdm
from trt_pose.coco import CocoDataset, CocoHumanPoseEval
def cmap_paf_loss(cmap,cmap_out,paf,paf_out,mask):
  cmap_mse = torch.mean(mask * (cmap_out - cmap)**2)
  paf_mse = torch.mean(mask * (paf_out - paf)**2)

  loss = cmap_mse + paf_mse
  return loss
def create_data_loaders(config):
    # train_dataset_kwargs = config["train_dataset"]
    # train_dataset_kwargs['transforms'] = torchvision.transforms.Compose([
    #         torchvision.transforms.ColorJitter(**config['color_jitter']),
    #         torchvision.transforms.ToTensor(),
    #         torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    
    test_dataset_kwargs = config["test_dataset"]
    test_dataset_kwargs['transforms'] = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if 'evaluation' in config:
        evaluator = CocoHumanPoseEval(**config['evaluation'])
    
    # train_dataset = CocoDataset(**train_dataset_kwargs)
    test_dataset = CocoDataset(**test_dataset_kwargs)
    
    part_type_counts = test_dataset.get_part_type_counts().float().cpu()
    part_weight = 1.0 / part_type_counts
    part_weight = part_weight / torch.sum(part_weight)
    paf_type_counts = test_dataset.get_paf_type_counts().float().cpu()
    paf_weight = 1.0 / paf_type_counts
    paf_weight = paf_weight / torch.sum(paf_weight)
    paf_weight /= 2.0
    
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     **config["train_loader"]
    # )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        **config["test_loader"]
    )
    return test_loader

def create_torch_model(MODEL_WEIGHTS,v=False):
    import json
    import trt_pose.coco
    import trt_pose.models
    from torchinfo import summary
    import torch


    with open('/trt_pose/tasks/human_pose/human_pose.json', 'r') as f:
        human_pose = json.load(f)

    topology = trt_pose.coco.coco_category_to_topology(human_pose)

    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).eval()

    #MODEL_WEIGHTS = '/trt_pose/tasks/human_pose/resnet18_baseline_att_224x224_A_epoch_249.pth'

    model.load_state_dict(torch.load(MODEL_WEIGHTS,map_location=torch.device('cpu')))

    if v:
        print(summary(model))

    return model


#device = torch.device("cuda")
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_config',
    default="/trt_pose/tasks/human_pose/experiments/resnet18_baseline_att_224x224_A.json",
   # help='Data set directory, when quant_mode=calib, it is for calibration, while quant_mode=test it is for evaluation'
)
parser.add_argument(
    '--model_dir',
    default="/path/to/trained_model/",
    help='Trained model file path. Download pretrained model from the following url and put it in model_dir specified path: https://download.pytorch.org/models/resnet18-5c106cde.pth'
)
parser.add_argument(
    '--config_file',
    default=None,
    help='quantization configuration file')
parser.add_argument(
    '--subset_len',
    default=None,
    type=int,
    help='subset_len to evaluate model, using the whole validation dataset if it is not set')
parser.add_argument(
    '--batch_size',
    default=32,
    type=int,
    help='input data batch size to evaluate model')
parser.add_argument('--quant_mode', 
    default='calib', 
    choices=['float', 'calib', 'test'], 
    help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
parser.add_argument('--fast_finetune', 
    dest='fast_finetune',
    action='store_true',
    help='fast finetune model before calibration')
parser.add_argument('--deploy', 
    dest='deploy',
    action='store_true',
    help='export xmodel for deployment')
parser.add_argument('--inspect', 
    dest='inspect',
    action='store_true',
    help='inspect model')
args, _ = parser.parse_known_args()

def load_data(train=True,
              data_dir='dataset/imagenet',
              batch_size=128,
              subset_len=None,
              sample_method='random',
              distributed=False,
              model_name='resnet18',
              **kwargs):

  #prepare data
  # random.seed(12345)
  traindir = data_dir + '/train'
  valdir = data_dir + '/val'
  train_sampler = None
  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  if model_name == 'inception_v3':
    size = 299
    resize = 299
  else:
    size = 224
    resize = 256
  if train:
    dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    if subset_len:
      assert subset_len <= len(dataset)
      if sample_method == 'random':
        dataset = torch.utils.data.Subset(
            dataset, random.sample(range(0, len(dataset)), subset_len))
      else:
        dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
    if distributed:
      train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **kwargs)
  else:
    dataset = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ]))
    if subset_len:
      assert subset_len <= len(dataset)
      if sample_method == 'random':
        dataset = torch.utils.data.Subset(
            dataset, random.sample(range(0, len(dataset)), subset_len))
      else:
        dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, **kwargs)
  return data_loader, train_sampler

class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self, name, fmt=':f'):
    self.name = name
    self.fmt = fmt
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions
    for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res

def evaluate(model, val_loader, loss_fn):

  model.eval()
  model = model.to(device)
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')
  total = 0
  Loss = 0

  

  for count, (image, cmap, paf, mask) in tqdm(enumerate(val_loader), total=len(val_loader)):
    image = image.to(device)
    cmap = cmap.to(device)
    paf = paf.to(device)
    mask = mask.to(device).float()

   
    mask = torch.ones_like(mask).to(device).float()
    
    cmap_out, paf_out = model(image)
    loss = loss_fn(cmap,cmap_out,paf,paf_out,mask)
    Loss += loss.item()
   # total += images.size(0)
    # acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
    # top1.update(acc1[0], images.size(0))
    # top5.update(acc5[0], images.size(0))
  return  Loss / len(val_loader)
  # for iteraction, (images, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
  #   images = images.to(device)
  #   labels = labels.to(device)
  #   #pdb.set_trace()
  #   outputs = model(images)
  #   loss = loss_fn(outputs, labels)
  #   Loss += loss.item()
  #   total += images.size(0)
  #   acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
  #   top1.update(acc1[0], images.size(0))
  #   top5.update(acc5[0], images.size(0))
  # return top1.avg, top5.avg, Loss / total

def quantization(title='optimize',
                 model_name='', 
                 file_path=''): 

  config_data = args.data_config
  quant_mode = args.quant_mode
  finetune = args.fast_finetune
  deploy = args.deploy
  batch_size = args.batch_size
  subset_len = args.subset_len
  inspect = args.inspect
  config_file = args.config_file
  if quant_mode != 'test' and deploy:
    deploy = False
    print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
  if deploy and (batch_size != 1 or subset_len != 1):
    print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
    batch_size = 1
    subset_len = 1

  model = create_torch_model(args.model_dir,v=True).cpu()
 # model.load_state_dict(torch.load(file_path))

  input = torch.randn([batch_size, 3, 224, 224])
  if quant_mode == 'float':
    quant_model = model
    if inspect:
      import sys
      from pytorch_nndct.apis import Inspector
      # create inspector
      # inspector = Inspector("0x603000b16013831") # by fingerprint
      inspector = Inspector("DPUCAHX8H_ISA2")  # by name
      # start to inspect
      inspector.inspect(quant_model, (input,), device=device)
      sys.exit()
  else:
    ## new api
    ####################################################################################
    quantizer = torch_quantizer(
        quant_mode, model, (input), device=device, quant_config_file=config_file)

    quant_model = quantizer.quant_model
    #####################################################################################

  # to get loss value after evaluation
 # loss_fn = torch.nn.CrossEntropyLoss().to(device)#CAMBIAR  
  loss_fn = cmap_paf_loss
  with open(config_data,'r') as f:
    config = json.load(f)

  test_loader = create_data_loaders(config)
#   val_loader, _ = load_data(
#       subset_len=subset_len,
#       train=False,
#       batch_size=batch_size,
#       sample_method='random',
#       data_dir=data_dir,
#       model_name=model_name)

  # fast finetune model or load finetuned parameter before test
  if finetune == True:
      ft_loader, _ = load_data(
          subset_len=5120,
          train=False,
          batch_size=batch_size,
          sample_method='random',
          data_dir=data_dir,
          model_name=model_name)
      if quant_mode == 'calib':
        quantizer.fast_finetune(evaluate, (quant_model, ft_loader, loss_fn))
      elif quant_mode == 'test':
        quantizer.load_ft_param()
   
  # record  modules float model accuracy
  # add modules float model accuracy here
  acc_org1 = 0.0
  acc_org5 = 0.0
  loss_org = 0.0

  #register_modification_hooks(model_gen, train=False)
  loss_gen = evaluate(quant_model, test_loader, loss_fn)

  # logging accuracy
  print('loss: %g' % (loss_gen))
 # print('top-1 / top-5 accuracy: %g / %g' % (acc1_gen, acc5_gen))

  # handle quantization result
  if quant_mode == 'calib':
    quantizer.export_quant_config()
  if deploy:
    quantizer.export_xmodel(deploy_check=False)
    quantizer.export_onnx_model()


if __name__ == '__main__':

  model_name = 'resnet18'
 # file_path = os.path.join(args.model_dir, model_name + '.pth')

  feature_test = ' float model evaluation'
  if args.quant_mode != 'float':
    feature_test = ' quantization'
    # force to merge BN with CONV for better quantization accuracy
    args.optimize = 1
    feature_test += ' with optimization'
  else:
    feature_test = ' float model evaluation'
  title = model_name + feature_test

  print("-------- Start {} test ".format(model_name))

  # calibration or evaluation
  quantization(
      title=title,
      model_name=model_name,
      file_path=args.model_dir)

  print("-------- End of {} test ".format(model_name))
