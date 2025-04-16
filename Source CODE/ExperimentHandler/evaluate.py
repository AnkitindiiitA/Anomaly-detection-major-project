import numpy as np
import torch
import wandb
import copy
import os
from avalanche.benchmarks.utils.avalanche_dataset import AvalancheSubset
from DataHandler.PaymentDataset import PaymentDataset
from UtilsHandler.UtilsHandler import UtilsHandler
from UtilsHandler.StrategyHandler import StrategyHandler
from UtilsHandler.BenchmarkHandler import BenchmarkHandler
from UtilsHandler.MetricHandler import MetricHandler


def evaluate_continual_checkpoint(experiment_parameters):
    # Initialize handlers
    uha = UtilsHandler()
    sha = StrategyHandler()
    bha = BenchmarkHandler()
    mha = MetricHandler()

    # Initialize payment dataset
    payment_ds = PaymentDataset(experiment_parameters['data_dir'])

    # Get index assignments for all experiences
    perc_matrix = bha.create_percnt_matrix(experiment_parameters)
    exp_assignments, samples_matrix = bha.get_exp_assignment(experiment_parameters, payment_ds, perc_matrix)

    # Get benchmark
    benchmark = bha.get_benchmark(experiment_parameters, payment_ds, exp_assignments)
    device = experiment_parameters['device']
    # Get Strategy
    strategy = sha.get_strategy(experiment_parameters, payment_ds)
    # === Load Checkpoint ===
    checkpoint_path = experiment_parameters["checkpoint_path"]
    if os.path.exists(checkpoint_path):
        print(f"Loading model from: {checkpoint_path}")
        # ✅ Correct way to load checkpoint
        state_dict = torch.load(checkpoint_path, map_location=device)
        strategy.model.load_state_dict(state_dict)

    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    # Initialize WandB
    run_name = experiment_parameters['run_name']
    log_wandb = experiment_parameters['wandb_proj'] != ''
    uha.init_wandb(experiment_parameters, run_name, log_wandb)
    output_path = os.path.join(experiment_parameters['outputs_path'], run_name)


    # ============================
    # Compute FPs and FNs in the Final Experience
    # ============================

    last_exp_id = len(benchmark.train_stream) - 1
    rec, info_rec = mha.compute_rec_ratio(strategy, benchmark.train_stream[last_exp_id].dataset,experiment_parameters)
    prec, info_prec = mha.compute_prec_ratio(strategy, benchmark.train_stream[last_exp_id].dataset,experiment_parameters)


    
    print("rec:",rec)
    print("prec:",prec)
