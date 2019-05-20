"""Light Neural Architecture Search."""
from __future__ import print_function

import json
import argparse
import subprocess
from paddle.fluid.contrib.slim.search import Searcher
from paddle.fluid.contrib.slim.search import SaController

NAS_FILTER_SIZE = [[18, 24, 30], [24, 32, 40], [48, 64, 80], [72, 96, 120],
                   [120, 160, 200]]
NAS_LAYERS_NUMBER = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [2, 3, 4], [2, 3, 4]]
NAS_KERNEL_SIZE = [3, 5]
NAS_FILTERS_MULTIPLIER = [3, 4, 5, 6]
NAS_SHORTCUT = [0, 1]
NAS_SE = [0, 1]


def get_bottleneck_params_list(var):
    """Get bottleneck_params_list from var.

    Args:
        var: list, variable list.

    Returns:
        list, bottleneck_params_list.
    """
    params_list = [
        1, 16, 1, 1, 3, 1, 0, \
        6, 24, 2, 2, 3, 1, 0, \
        6, 32, 3, 2, 3, 1, 0, \
        6, 64, 4, 2, 3, 1, 0, \
        6, 96, 3, 1, 3, 1, 0, \
        6, 160, 3, 2, 3, 1, 0, \
        6, 320, 1, 1, 3, 1, 0, \
    ]
    for i in range(5):
        params_list[i * 7 + 7] = NAS_FILTERS_MULTIPLIER[var[i * 6]]
        params_list[i * 7 + 8] = NAS_FILTER_SIZE[i][var[i * 6 + 1]]
        params_list[i * 7 + 9] = NAS_LAYERS_NUMBER[i][var[i * 6 + 2]]
        params_list[i * 7 + 11] = NAS_KERNEL_SIZE[var[i * 6 + 3]]
        params_list[i * 7 + 12] = NAS_SHORTCUT[var[i * 6 + 4]]
        params_list[i * 7 + 13] = NAS_SE[var[i * 6 + 5]]
    return params_list


def get_reward_command(var):
    """A command to get reward.

    Args:
        var: str, a string that represents variable list.

    Returns:
        list, a list of strings represents the command.
    """
    var = json.loads(var)
    bottleneck_params_list = get_bottleneck_params_list(var)
    print(bottleneck_params_list)
    return [
        'python', 'light_nas_main.py', '--bottleneck_params_list={}'.format(
            str(bottleneck_params_list).replace(' ', ''))
    ]


def run_final(var):
    """Run final training.

    Args:
        var: str, a string that represents variable list.

    Returns:
        list, a list of strings represents the command.
    """
    bottleneck_params_list = get_bottleneck_params_list(var)
    subprocess.call([
        'python', '-u', 'light_nas_train.py',
        '--bottleneck_params_list={}'.format(
            str(bottleneck_params_list).replace(' ', ''))
    ])


def main():
    """main."""
    parser = argparse.ArgumentParser()
    # controller related arguments
    parser.add_argument(
        '--range_table',
        default=[
            4, 3, 3, 2, 2, 2, 4, 3, 3, 2, 2, 2, 4, 3, 3, 2, 2, 2, 4, 3, 3, 2, 2,
            2, 4, 3, 3, 2, 2, 2
        ],
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
    # NAS related arguments
    parser.add_argument(
        '--max_iterations', default=300, type=int, help='maximum iterations.')
    parser.add_argument(
        '--init_var',
        default=[
            3, 1, 1, 0, 1, 0, 3, 1, 1, 0, 1, 0, 3, 1, 1, 0, 1, 0, 3, 1, 1, 0, 1,
            0, 3, 1, 1, 0, 1, 0
        ],
        type=list,
        help='init variable list.')
    parser.add_argument(
        '--init_reward', default=None, type=float, help='init reward.')
    parser.add_argument('--verbose', default=True, type=bool, help='verbose.')
    args = parser.parse_args()
    controller = SaController(args.range_table, args.reduce_rate,
                              args.init_temperature)
    searcher = Searcher(controller, get_reward_command, args.verbose)
    if not args.init_var:
        init_var = controller.generate_init_var()
    else:
        init_var = args.init_var
    if not args.init_reward:
        init_reward = searcher.get_reward(str(init_var))
    else:
        init_reward = args.init_reward
    var_max, reward_max = searcher.search(args.max_iterations, init_var,
                                          init_reward)
    print('Max reward: {}'.format(reward_max))
    print('Best var: {}'.format(var_max))
    run_final(var_max)


if __name__ == '__main__':
    main()
