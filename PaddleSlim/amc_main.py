import sys, os
import json
import argparse
from paddle.fluid.contrib.slim.search import Searcher
from paddle.fluid.contrib.slim.search import SaController


def get_reward_command(var):
    """A command to get reward.
    Args:
        var: a string that represents variable list.
    Returns:
        list, a list of strings represents the command.
    """
    var = json.loads(var)
    return [
        'python', 'amc_train.py', '--batch_size', '128', '--method', 'reward',
        '--ratio_list', str(var).replace(' ', '')
    ]


def get_flops_command(var):
    """A command to get flops.
    Args:
        var: a string that represents variable list.
    Returns:
        list, a list of strings represents the command.
    """
    return [
        'python', 'amc_train.py', '--batch_size', '128', '--method', 'flops',
        '--ratio_list', str(var).replace(' ', '')
    ]


def main():
    """main.
    """
    parser = argparse.ArgumentParser()
    # controller related arguments
    parser.add_argument(
        '--range_table',
        default=[40] * 13,
        type=list,
        help='variable range table.')
    parser.add_argument(
        '--reduce_rate',
        default=0.9,
        type=float,
        help='reduce rate for sa controller.')
    parser.add_argument(
        '--init_temperature',
        default=100,
        type=float,
        help='init temperature for sa controller.')
    parser.add_argument(
        '--constrain',
        default=1,
        type=int,
        help='flops constrain for sa controller.')
    parser.add_argument(
        '--max_threshold',
        default=0.50,
        type=float,
        help='max threshold of constrain for sa controller.')
    parser.add_argument(
        '--min_threshold',
        default=0.48,
        type=float,
        help='min threshold of constrain for sa controller.')
    parser.add_argument(
        '--max_iter_number',
        default=300,
        type=int,
        help='max iter number of constrain for sa controller.')
    # NAS related arguments
    parser.add_argument(
        '--max_iterations', default=2000, type=int, help='maximum iterations.')
    parser.add_argument(
        '--init_var', default=None, type=list, help='init variable list.')
    parser.add_argument(
        '--init_reward', default=0, type=float, help='init reward.')
    parser.add_argument('--verbose', default=True, type=bool, help='verbose.')
    args = parser.parse_args()
    controller = SaController(args.range_table, args.reduce_rate,
                              args.init_temperature, args.constrain,
                              get_flops_command, args.max_thres, args.min_thres,
                              args.max_iter_number)
    amc = Searcher(controller, get_reward_command, args.verbose)
    if not args.init_var:
        init_var = controller.generate_init_var()
    else:
        init_var = args.init_var
    if not args.init_reward:
        init_reward = amc.get_reward(str(init_var))
    else:
        init_reward = args.init_reward
    var_max, reward_max = amc.search(args.max_iterations, init_var, init_reward)
    print('Max reward: {}'.format(reward_max))
    print('Best var: {}'.format(var_max))

    print("begin training")
    command = 'python amc_train.py --ratio_list ' + str(var_max).replace(
        ' ', '') + ' --method train'
    os.system(command)
    print("finish training")


if __name__ == '__main__':
    main()
