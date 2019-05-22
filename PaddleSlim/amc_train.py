from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import sys
import json
import logging
import re
import paddle
import models
import argparse
import functools
import paddle.fluid as fluid
import reader
from utility import add_arguments, print_arguments

from paddle.fluid.contrib.slim import Compressor, Context
from paddle.fluid.contrib.slim.graph import *

logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s')
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  256,                 "Minibatch size.")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('class_dim',        int,  1000,                "Class number.")
add_arg('image_shape',      str,  "3,224,224",         "Input image size")
add_arg('model',            str,  "MobileNet",         "Set the network to use.")
add_arg('pretrained_model', str,  "./pretrain/MobileNetV1_pretrained",                "Whether to use pretrained model.")
add_arg('teacher_model',    str,  None,          "Set the teacher network to use.")
add_arg('teacher_pretrained_model', str,  None,                "Whether to use pretrained model.")
add_arg('compress_config',  str,  "./configs/filter_pruning_uniform.yaml",                 "The config file for compression with yaml format.")
add_arg('ratio_list',       str,  None,                "The pruning ratio of each pruned param.")
add_arg('method',           str,  None,                "The method.")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]
os.system('export CUDA_VISIBLE_DEVICES=0,1,2,3')


def get_result(args):
    image_shape = [int(m) for m in args.image_shape.split(",")]

    assert args.model in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)
    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    # model definition
    model = models.__dict__[args.model]()

    if args.model is "GoogleNet":
        out0, out1, out2 = model.net(input=image, class_dim=args.class_dim)
        cost0 = fluid.layers.cross_entropy(input=out0, label=label)
        cost1 = fluid.layers.cross_entropy(input=out1, label=label)
        cost2 = fluid.layers.cross_entropy(input=out2, label=label)
        avg_cost0 = fluid.layers.mean(x=cost0)
        avg_cost1 = fluid.layers.mean(x=cost1)
        avg_cost2 = fluid.layers.mean(x=cost2)
        avg_cost = avg_cost0 + 0.3 * avg_cost1 + 0.3 * avg_cost2
        acc_top1 = fluid.layers.accuracy(input=out0, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out0, label=label, k=5)
    else:
        out = model.net(input=image, class_dim=args.class_dim)
        cost = fluid.layers.cross_entropy(input=out, label=label)
        avg_cost = fluid.layers.mean(x=cost)
        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
    val_program = fluid.default_main_program().clone()

    opt = fluid.optimizer.Momentum(
        momentum=0.9,
        learning_rate=fluid.layers.piecewise_decay(
            boundaries=[5000 * 30, 5000 * 60, 5000 * 90],
            values=[0.1, 0.01, 0.001, 0.0001]),
        regularization=fluid.regularizer.L2Decay(4e-5))

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if args.pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(args.pretrained_model, var.name))

        fluid.io.load_vars(exe, args.pretrained_model, predicate=if_exist)

    val_reader = paddle.batch(reader.val(), batch_size=args.batch_size)
    val_feed_list = [('image', image.name), ('label', label.name)]
    val_fetch_list = [('acc_top1', acc_top1.name), ('acc_top5', acc_top5.name)]

    train_reader = paddle.batch(
        reader.train(), batch_size=args.batch_size, drop_last=True)
    train_feed_list = [('image', image.name), ('label', label.name)]
    train_fetch_list = [('loss', avg_cost.name)]

    teacher_programs = []
    distiller_optimizer = None
    if args.teacher_model:
        teacher_model = models.__dict__[args.teacher_model]()
        # define teacher program
        teacher_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(teacher_program, startup_program):
            img = teacher_program.global_block()._clone_variable(
                image, force_persistable=False)
            predict = teacher_model.net(img,
                                        class_dim=args.class_dim,
                                        conv1_name='res_conv1',
                                        fc_name='res_fc')
        exe.run(startup_program)
        assert args.teacher_pretrained_model and os.path.exists(
            args.teacher_pretrained_model
        ), "teacher_pretrained_model should be set when teacher_model is not None."

        def if_exist(var):
            return os.path.exists(
                os.path.join(args.teacher_pretrained_model, var.name))

        fluid.io.load_vars(
            exe,
            args.teacher_pretrained_model,
            main_program=teacher_program,
            predicate=if_exist)

        distiller_optimizer = opt
        teacher_programs.append(teacher_program.clone(for_test=True))

    com_pass = Compressor(
        place,
        fluid.global_scope(),
        fluid.default_main_program(),
        train_reader=train_reader,
        train_feed_list=train_feed_list,
        train_fetch_list=train_fetch_list,
        eval_program=val_program,
        eval_reader=val_reader,
        eval_feed_list=val_feed_list,
        eval_fetch_list=val_fetch_list,
        teacher_programs=teacher_programs,
        train_optimizer=opt,
        distiller_optimizer=distiller_optimizer)
    com_pass.config(args.compress_config)

    train_graph = GraphWrapper(
        fluid.default_main_program(),
        in_nodes=train_feed_list,
        out_nodes=train_fetch_list)
    eval_graph = GraphWrapper(
        val_program, in_nodes=val_feed_list, out_nodes=val_fetch_list)
    context = Context(
        place=place,
        scope=fluid.global_scope(),
        train_graph=train_graph,
        train_reader=train_reader,
        eval_graph=eval_graph,
        eval_reader=val_reader,
        teacher_graphs=teacher_programs,
        train_optimizer=opt,
        distiller_optimizer=distiller_optimizer)

    com_pass._init_model(context)
    if not context.optimize_graph:
        if context.train_optimizer:
            context.train_optimizer._name = 'train_opt'
            context.optimize_graph = context.train_graph.get_optimize_graph(
                context.train_optimizer, context.place, context.scope)
        else:
            context.optimize_graph = context.train_graph

    context, com_pass.strategies = com_pass._load_checkpoint(context)

    ratio_list = json.loads(
        args.
        ratio_list) if not args.ratio_list is None else [0 for i in range(13)]
    ratio_list = [i * 0.01 for i in ratio_list]
    print("ratio_list: {}".format(ratio_list))

    for strategy in com_pass.strategies:
        pruned_params = []
        for param in context.eval_graph.all_parameters():
            if re.match(strategy.pruned_params, param.name()):
                pruned_params.append(param.name())
        strategy._prune_parameters(context.optimize_graph, context.scope,
                                   pruned_params, ratio_list, context.place)
        strategy._prune_graph(context.eval_graph, context.optimize_graph)
        context.optimize_graph.update_groups_of_conv()
        context.eval_graph.update_groups_of_conv()

        context.optimize_graph.compile()
        context.eval_graph.compile(
            for_parallel=False, for_test=True)  # to update the compiled program

    if args.method == 'flops':
        flops = context.eval_graph.flops()
        model_size = context.eval_graph.numel_params()
        print("{} {}".format(flops, model_size))
    elif args.method == 'reward':
        com_pass._eval(context)
        print("{}".format(context.eval_results['acc_top1'][0]))
    else:
        start = context.epoch_id
        for epoch in range(start, com_pass.epoch):
            context.epoch_id = epoch
            com_pass._train_one_epoch(context)
            if com_pass.eval_epoch and epoch % com_pass.eval_epoch == 0:
                com_pass._eval(context)
            com_pass._save_checkpoint(context)


def main():
    args = parser.parse_args()
    print_arguments(args)
    get_result(args)


if __name__ == '__main__':
    main()
