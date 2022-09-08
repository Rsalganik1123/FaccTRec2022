
import os 
import sys

# from NodeClassification.GS.main import * 
# from NodeClassification.utils.launchers.NC_args import *
from LinkPrediction.GS.main import * 
from LinkPrediction.utils.launchers.LP_args import *  
from orion.client import report_objective, build_experiment 

def run():
    args = get_args()
    print(args)
    SAGE_loop_LP(args)


def search(): 
    experiment = build_experiment(
        name="GS_TEST",
        space={
            "lr": "uniform(0.001, 0.1)",
            "pretrain_epochs": "uniform(30, 100, discrete=True)",
            "epochs": "uniform(15, 60, discrete=True)",
            "dropout": "uniform(0.001, 0.01)",
            "weight_decay": "uniform(0.001, 0.01)",
        },
        storage={
            "type": "legacy",
            "database": {"type": "pickleddb", "host": "db2.pkl"},
        },
        algorithms="tpe",
        max_trials=10,
    )
    trials = 0 
    while not experiment.is_done:
        trial = experiment.suggest()
        print("STARTING TRIAL:{} - {} ".format(trials, trial.params)) 
        lr = trial.params["lr"]
        epochs = trial.params['epochs']
        params = {'exp_name': 'GS', 'dataset': 'Cora', 'similarity': 'jaccard',  'top_k': 10,  'fine_tune': False,
            'hidden': 256, 'batch_size': 32, 'n_components': 200, 'layer_neighbors': [15,15]}  # set fixed parameters.
        params.update(trial.params) # update with the sampled parameters.
        trials += 1 

        print(params)
        
        # execute the function
        best_val_acc = SAGE_loop(params)
        
        experiment.observe(trial, results=[dict(type="objective", name="neg_val_acc", value=best_val_acc)])

run() 

# search() 
