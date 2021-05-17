import torch, argparse

def parse_args():
	parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('dir', help='test config file path')
    parser.add_argument('new_dir', help='new test config file path')
    args = parser.parse_args()
    return args

args = parse_args()
a = torch.load(args.dir)
a_weight = a['state_dict']
torch.save(a_weight, args.new_dir)