#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Testing for a Task in Neural Turing Machines."""

import argparse
import json
import logging
import time
import random
import re
import sys

import attr
import argcomplete
import torch
from torch.nn.parallel import data_parallel
import numpy as np


LOGGER = logging.getLogger(__name__)

from tasks.copytask import CopyTaskModelTraining, CopyTaskParams
from tasks.repeatcopytask import RepeatCopyTaskModelTraining, RepeatCopyTaskParams

TASKS = {
    'copy': (CopyTaskModelTraining, CopyTaskParams),
    'repeat-copy': (RepeatCopyTaskModelTraining, RepeatCopyTaskParams)
}

# Default values for program arguments
REPORT_INTERVAL = 200

def get_ms():
    """Returns the current time in miliseconds."""
    return time.time() * 1000

def init_seed(seed=None):
    """Seed the RNGs for predicatability/reproduction purposes."""
    if seed is None:
        seed = int(get_ms() // 1000)

    LOGGER.info("Using seed=%d", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
    random.seed(seed)

def progress_clean():
    """Clean the progress bar."""
    print ("\r{}".format(" " * 80), end='\r')


def progress_bar(batch_num, report_interval, last_loss):
    """Prints the progress until the next report."""
    progress = (((batch_num-1) % report_interval) + 1) / report_interval
    fill = int(progress * 40)
    print("\r[{}{}]: {} (Loss: {:.4f})".format(
        "=" * fill, " " * (40 - fill), batch_num, last_loss), end='')


def evaluate_model(model, args):
    num_batches = model.params.num_batches
    batch_size = model.params.batch_size

    LOGGER.info("Evaluating model for %d batches (batch_size=%d)...",
                num_batches, batch_size)

    losses = []
    costs = []
    start_ms = get_ms()

    #seq_len = 80
    #_, x, y = next(iter(model.dataloader(1, 1, 8, seq_len, seq_len)))
    #for batch_num in range(500, 10500, 500):
    for batch_num, x, y in model.dataloader:
        if args.use_cuda and torch.cuda.is_available():
            use_cuda = True
            x = x.cuda()
            y = y.cuda()
        else:
            use_cuda = False

        with torch.no_grad():
            result = evaluate_batch(model.net, model.criterion, x, y, args.use_cuda)
        loss = result['loss']
        losses += [loss]
        costs += [result['cost']]

        # Update the progress bar
        if args.show_progress:
            progress_bar(batch_num, args.report_interval, loss)

        # Report
        if batch_num % args.report_interval == 0:
            mean_loss = np.array(losses[-args.report_interval:]).mean()
            mean_cost = np.array(costs[-args.report_interval:]).mean()
            mean_time = int(((get_ms() - start_ms) / args.report_interval) / batch_size)
            if args.show_progress:
                progress_clean()
            LOGGER.info("Batch %d Loss: %.6f Cost: %.2f Time: %d ms/sequence",
                        batch_num, mean_loss, mean_cost, mean_time)
            start_ms = get_ms()

    LOGGER.info("Done evaluating.")

def evaluate_batch(net, criterion, X, Y, use_cuda):
    """Evaluate a single batch (without training)."""
    inp_seq_len = X.size(0)
    outp_seq_len, batch_size, _ = Y.size()

    # New sequence
    net.init_sequence(batch_size, use_cuda)

    # Feed the sequence + delimiter
    states = []
    for i in range(inp_seq_len):
        if use_cuda and torch.cuda.is_available():
            o, state = data_parallel(net, X[i])
        else:
            o, state = net(X[i])
        states += [state]

    # Read the output (no input given)
    y_out = []
    for i in range(outp_seq_len):
        if use_cuda and torch.cuda.is_available():
            o, state  = data_parallel(net, X[i])
        else:
            o, _ = net(X[i])
        states += [state]
        y_out += [o]
    y_out = torch.cat(y_out, dim=0).unsqueeze(1)

    loss = criterion(y_out, Y)

    y_out_binarized = y_out.clone().data
    for i in y_out_binarized:
        for j in i:
            for k in j:
                k = 0 if k < 0.5 else 1

    # The cost is the number of error bits per sequence
    cost = torch.sum(torch.abs(y_out_binarized - Y.data))

    result = {
        'loss': loss.data.item(),
        'cost': cost / batch_size,
        'y_out': y_out,
        'y_out_binarized': y_out_binarized,
        'states': states
    }

    return result


def init_arguments():
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--task', action='store', choices=list(TASKS.keys()), default='copy',
                        help="Choose the task to test (default: copy)")
    parser.add_argument('--model', action='store', type=str, help="Provide the model path to evaluate")
    parser.add_argument('-p', '--param', action='append', default=[],
                        help='Override model params. Example: "-pbatch_size=4 -pnum_heads=2"')
    parser.add_argument('--report-interval', type=int, default=REPORT_INTERVAL,
                        help="Reporting interval")
    parser.add_argument('--gpu', dest='use_cuda', default=False, action='store_true', help='Use the GPU')
    parser.add_argument('--time', dest='time', default=False, action='store_true', help='Time the execution')
    parser.add_argument('--show_progress', dest='show_progress', default=False, action='store_true', help='Print out progress bar')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    return args


def update_model_params(params, update):
    """Updates the default parameters using supplied user arguments."""

    update_dict = {}
    for p in update:
        m = re.match("(.*)=(.*)", p)
        if not m:
            LOGGER.error("Unable to parse param update '%s'", p)
            sys.exit(1)

        k, v = m.groups()
        update_dict[k] = v

    try:
        params = attr.evolve(params, **update_dict)
    except TypeError as e:
        LOGGER.error(e)
        LOGGER.error("Valid parameters: %s", list(attr.asdict(params).keys()))
        sys.exit(1)

    return params

def load_model(args):
    model_cls, params_cls = TASKS[args.task]
    params = params_cls()
    model_file = args.model

    params = update_model_params(params, args.param)
    LOGGER.info(params)

    model = model_cls(params=params, cuda=args.use_cuda, time=args.time)

    if model_file:
        LOGGER.info("Loading Model: '%s'", model_file)
        model.net.load_state_dict(torch.load(model_file))
        LOGGER.info("Loaded Model: '%s'", model_file)
    else:
        LOGGER.info("No Pretrained Model Found!")

    return model


def init_logging():
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s]  %(message)s',
                        level=logging.DEBUG)


def main():
    torch.set_num_threads(4)
    torch.set_printoptions(threshold=5000)
    init_logging()

    # Initialize arguments
    args = init_arguments()
    # Initialize random
    #init_seed(args.seed)

    # Load the model
    model = load_model(args)

    LOGGER.info("Total number of parameters: %d", model.net.calculate_num_params())
    evaluate_model(model, args)

if __name__ == '__main__':
    main()
