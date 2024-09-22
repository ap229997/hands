import common.comet_utils as comet_utils
import common.tb_utils as tb_utils
from src.parsers.parser import construct_args

args = construct_args()
if args.logger == 'comet':
    experiment, args = comet_utils.init_experiment(args)
elif args.logger == 'tensorboard':
    args = tb_utils.init_experiment(args)
save_keys = list(args.keys())
save_keys.remove('experiment')
if args.logger == 'comet':
    comet_utils.save_args(args, save_keys=save_keys)
elif args.logger == 'tensorboard':
    tb_utils.save_args(args, save_keys=save_keys)
