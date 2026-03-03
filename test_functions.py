import argparse
import datasets
import utilites as utils
from models import BinaryLogisticRegression,NN
import main
import pickle
import os
import torch
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Run federated learning experiments.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--method',     type=str,   default='FairFed_w_FairBatch',
                    choices=['FairFed_w_FairBatch', 'FedAvg', 'MinMax', 'Central',
                             'KRTWD', 'KRTD', 'FairFed_w_FairBatch_kernel'],
                    help='Federated learning method to run.')
parser.add_argument('--model',      type=str,   default='LR',
                    choices=['LR', 'NN'],
                    help='Model architecture.')
parser.add_argument('--dataset',    type=str,   default='ADULT',
                    choices=['ADULT', 'COMPAS'],
                    help='Dataset to use.')
parser.add_argument('--dist',       type=str,   default='Dirichlet',
                    choices=['IID', 'Non-IID', 'Dirichlet'],
                    help='Client data distribution.')
parser.add_argument('--num_rounds', type=int,   default=200,
                    help='Number of federated training rounds.')
parser.add_argument('--num_sims',   type=int,   default=5,
                    help='Number of independent simulations (seeds).')
parser.add_argument('--step_size',  type=float, default=0.01,
                    help='Global / client learning rate.')
parser.add_argument('--fb_lr',      type=float, default=0.005,
                    help='FairBatch weight-update learning rate.')
parser.add_argument('--alpha',      type=float, default=0.1,
                    help='Dirichlet concentration parameter.')
parser.add_argument('--fairness',   type=float, default=None,
                    help='Fairness weight (for KRTWD, KRTD, Central). '
                         'If not set, the method-specific default grid is used.')
args = parser.parse_args()

# Map parsed args to the variables the rest of the script uses
dataset  = [args.dataset]
model    = [args.model]
methods  = [args.method]
dist     = [args.dist]
numsims  = args.num_sims

print("=" * 60)
print(f"  method     : {args.method}")
print(f"  model      : {args.model}")
print(f"  dataset    : {args.dataset}")
print(f"  dist       : {args.dist}")
print(f"  num_rounds : {args.num_rounds}")
print(f"  num_sims   : {args.num_sims}")
print(f"  step_size  : {args.step_size}")
print(f"  fb_lr      : {args.fb_lr}")
print(f"  alpha      : {args.alpha}")
print(f"  fairness   : {args.fairness}")
print("=" * 60)

# Fairness weight grid (used when --fairness is not specified)
fairness_params = {
    'KRTWD': list(np.linspace(20, 1000, 20)),
    'KRTD': [100],
    'FairFed_w_FairBatch_kernel': [0],
    'Central': [10**-8, 10**-9, 10**-10, 10**-11, 10**-12, 10**-13],
}

for i in range(numsims):
    print(f"**************************Simulation is {i}***************")
    for distribution in dist:
        print(f"Distribution is {distribution}")
        for d in dataset:
            for mdl in model:
                for m in methods:
                    # --fairness on command line overrides the default grid
                    if args.fairness is not None:
                        fairness_weights = [args.fairness]
                    elif m in fairness_params:
                        fairness_weights = fairness_params[m]
                    else:
                        fairness_weights = [None]

                    for fairness_weight in fairness_weights:
                        print(f'Fairness weight is : {fairness_weight}')
                        if distribution == 'IID':
                            filename = f"{m}_{mdl}_{d}_{distribution}_{fairness_weight}_test_results.pickle"
                        else:
                            filename = f"{m}_{mdl}_{d}_{distribution}_{fairness_weight}_test_results_90_10.pickle"

                        params = {'fairness': fairness_weight} if fairness_weight is not None else None
                        results, results_std = main.simulation_runs(
                            m, mdl, distribution, d, numsims, params,
                            step_size=args.step_size,
                            fb_lr=args.fb_lr,
                            num_rounds=args.num_rounds,
                            alpha=args.alpha,
                        )
                        with open(filename, "wb") as file:
                            pickle.dump(results, file)
                            pickle.dump(results_std, file)
                            pickle.dump(d, file)
                            pickle.dump(mdl, file)
                            pickle.dump(distribution, file)
                            if fairness_weight is not None:
                                pickle.dump(fairness_weight, file)



