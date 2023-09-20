import argparse

__all__ = ['parse_cmd_args']


def create_parser():
    parser = argparse.ArgumentParser(description='Semi-supevised Training --PyTorch ')

    # Log and save
    parser.add_argument('--print-freq', default=20, type=int, metavar='N', help='display frequence (default: 20)')
    parser.add_argument('--save-freq', default=0, type=int, metavar='EPOCHS', help='checkpoint frequency(default: 0)')
    parser.add_argument('--save-dir', default='./checkpoints', type=str, metavar='DIR')
    parser.add_argument('--data-root', default='./data-local/cifar10/', type=str, metavar='DIR')

    # Data
    parser.add_argument('--dataset', type=str, default='cifar10', metavar='DATASET')
    parser.add_argument('--code-bits', type=int, default=12, help="hash code length")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--num-labels', type=int, default=5000, metavar='N', help='number of labeled samples')
    parser.add_argument('--num-unlabel', type=int, default=54000, help='number of unlabeled samples')
    parser.add_argument('--num-query', type=int, default=1000, help='number of query samples')
    parser.add_argument('--num-classes', type=int, default=10, help='number of classes')
    parser.add_argument('--sup-batch-size', default=100, type=int, metavar='N', help='batch size for supervised data (default: 100)')
    parser.add_argument('--usp-batch-size', default=100, type=int, metavar='N', help='batch size for unsupervised data (default: 100)')

    # Data pre-processing
    # Architecture
    parser.add_argument('--drop-ratio', default=0.5, type=float, help='ratio of dropout (default: 0)')

    # Optimization
    parser.add_argument('--seed', type=int, default=42, help="manual seed")
    parser.add_argument('--random', type=bool, default=True, help="not use random seed")
    parser.add_argument('--epochs', type=int, metavar='N', help='number of total training epochs')
    parser.add_argument('--optim', default="sgd", type=str, metavar='TYPE', choices=['sgd', 'adam'])
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--nesterov', default=True, type=str2bool, metavar='BOOL', help='use nesterov momentum (default: False)')
    parser.add_argument('--weight-decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    
    # LR schecular
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='max learning rate (default: 0.1)')
    parser.add_argument('--lr-scheduler', default="cos", type=str, choices=['cos', 'multistep', 'exp-warmup','cos-warmup', 'none'])
    parser.add_argument('--min-lr',  default=1e-4, type=float, metavar='LR', help='minimum learning rate (default: 1e-4)')
    parser.add_argument('--steps', type=int, nargs='+', metavar='N', help='decay steps for multistep scheduler')
    parser.add_argument('--gamma', type=float, help='factor of learning rate decay')
    parser.add_argument('--rampup-length', default=80, type=int, metavar='EPOCHS', help='length of the ramp-up')
    parser.add_argument('--rampdown-length', default=50, type=int, metavar='EPOCHS', help='length of the ramp-down')


    # Fixmatch
    parser.add_argument('--threshold', default=0.95,  type=float, metavar='W', help='threshold for confident predictions in Fixmatch')
    
    # MeanTeacher-based method
    parser.add_argument('--ema-decay', type=float, metavar='W', help='ema weight decay')


    # Opt for loss
    parser.add_argument('--usp-weight', default=1.0, type=float, metavar='W', help='the upper of unsuperivsed weight (default: 1.0)')
    parser.add_argument('--cons-weight', default=1.0, type=float, metavar='W', help='the upper of unsuperivsed weight (default: 1.0)')
    parser.add_argument('--contras-weight', default=1.0, type=float, metavar='W', help='the upper of unsuperivsed weight (default: 1.0)')

    parser.add_argument('--weight-rampup', default=30, type=int, metavar='EPOCHS', help='the length of rampup weight (default: 30)')

    # temperature
    parser.add_argument('--contras-t1', default=2.5, type=float, metavar='W', help='temperature of SupConLoss')
    parser.add_argument('--contras-t2', default=2.5, type=float, metavar='W', help='temperature of image_wise_loss')
    parser.add_argument('--consis-t', default=2.0, type=float, metavar='W', help='temperature of consistency')

    return parser


def parse_commandline_args():
    return create_parser().parse_args()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
