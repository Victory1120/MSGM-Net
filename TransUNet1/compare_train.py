import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
# from networks.z_model_v7.z_model_v7.vit_seg_modeling import VisionTransformer as ViT_seg
# from networks.z_model_v7.z_model_v7.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
# from networks.xiaorong.model_1.model_1.vit_seg_modeling import VisionTransformer as ViT_seg
# from networks.xiaorong.model_1.model_1.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
# from networks.z_model_v1.vit_seg_modeling import VisionTransformer as ViT_seg
# from networks.z_model_v1.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
# from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
# from networks.Defconv.DefEDNet import DefED_Net
from trainer import trainer_synapse
from networks.unet import UNet
# from networks.MultiResUNet import MultiResUnet
# from networks.attunet import AttU_Net
# from networks.swinunet.config import get_config
# from networks.swinunet.vision_transformer import SwinUnet as ViT_seg
# from networks.LeViT import Build_LeViT_UNet_384
# from networks.defConv.vit_seg_modeling import VisionTransformer as ViT_seg
# from networks.defConv.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
# from networks.our_LeViT384 import Build_LeViT_UNet_384
# from networks.missformer.MISSFormer import MISSFormer
from networks.CPFNet.model.BaseNet import CPFNet

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/sys123456/E21201096/datas/brats2D/1850train', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='compare_brats', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/new_brats', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=1800, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=32, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                                          default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--cfg', type=str, default='/home/sys123456/E21201096/zhou_pytorch/Z_TransUNet/configs/swin_tiny_patch4_window7_224_lite.yaml',
                    metavar="FILE", help='path to config file' )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
args = parser.parse_args()

#################################
def netParams(model):
    """
    computing total network parameters
    args:
       model: model
    return: the number of parameters
    """
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters

if __name__ == "__main__":

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': '/home/sys123456/E21201096/datas/Sypnase_2D/Synapse/train_npz',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
        'SegTHOR': {
            'root_path': '/home/sys123456/E21201096/datas/SegTHOR_2D/crop_train',
            'list_dir': './lists/lists_SegTHOR',
            'num_classes': 5,
        },
        'Baoceng': {
            'root_path': '/home/sys123456/E21201096/datas/baoceng_2D/2D_train',
            'list_dir': './lists/lists_baoceng',
            'num_classes': 3,
        },
        'ACDC': {
            'root_path': '/home/sys123456/E21201096/datas/ACDC_2D/train',
            'list_dir': './lists/lists_ACDC',
            'num_classes': 4,
        },
        'compare_brats': {
            'root_path': '/home/sys123456/E21201096/datas/brats2D/1850train',
            'list_dir': './lists/new_brats',
            'num_classes': 4,
    },

    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = False
    # args.is_pretrain = False
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    # print('snapshot_path:{}'.format(snapshot_path))

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    # Unet模型
    # net = UNet(1,3,32).cuda()  # 3可变
    # Atten_unet
    # net = AttU_Net(1,5).cuda()
    # swin_unet
    # config = get_config(args)
    # MultiResUNet
    # net = MultiResUnet().cuda()
    # net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
    # transunet+ours
    # net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    # net.load_from(weights=np.load(config_vit.pretrained_path))
    # defconv
    # net = DefED_Net(num_classes=args.num_classes).cuda()

        # 3D模型-Vnet
    # net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    # # net.load_from(weights=np.load(config_vit.pretrained_path))
    # LeViT-UNet
    # net = Build_LeViT_UNet_384(num_classes=4, pretrained=False).cuda()
    # our_LeViT-UNet
    # net = Build_LeViT_UNet_384(num_classes=5, pretrained=True).cuda()
    # MISSFormer
    # net = MISSFormer(num_classes=args.num_classes).cuda()
    # CPFNet
    net = CPFNet(out_planes=args.num_classes).cuda()

    total_paramters = netParams(net)
    print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))

    trainer = {'compare_brats': trainer_synapse,}
trainer[dataset_name](args, net, snapshot_path)