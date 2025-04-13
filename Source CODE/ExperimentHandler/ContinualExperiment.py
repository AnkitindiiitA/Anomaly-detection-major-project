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


def run_continual_experiment(experiment_parameters):
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

    # Get Strategy
    strategy = sha.get_strategy(experiment_parameters, payment_ds)

    # Initialize WandB
    run_name = experiment_parameters['run_name']
    log_wandb = experiment_parameters['wandb_proj'] != ''
    uha.init_wandb(experiment_parameters, run_name, log_wandb)
    output_path = os.path.join(experiment_parameters['outputs_path'], run_name)

    # Global iterator: starts from 0
    global_iter = 0

    if log_wandb:
        data_perc_table = wandb.Table(columns=[f"{dept_id}" for dept_id in experiment_parameters["dept_ids"]],
                                      data=perc_matrix)
        wandb.log({"Data Percentage Matrix": data_perc_table}, step=global_iter)

        data_samples_table = wandb.Table(columns=[f"{dept_id}" for dept_id in experiment_parameters["dept_ids"]],
                                      data=samples_matrix)
        wandb.log({"Data Samples Matrix": data_samples_table}, step=global_iter)



    # iterate through all experiences (tasks) and train the model for each experience
    for exp_id, exp in enumerate(benchmark.train_stream):
        # ============================
        # Train and evaluate on the current experience
        # ============================
        res_train = strategy.train(exp)
        loss_train_exp = res_train[f"Loss_Epoch/train_phase/train_stream/Task000"]

        # eval loss for the current experiment
        res_eval = strategy.eval(benchmark.train_stream[:exp_id+1])
        loss_eval_exp = res_eval[f"Loss_Exp/eval_phase/train_stream/Task000/Exp{exp_id:03d}"]
        loss_eval_exp_allseen = res_eval[f"Loss_Stream/eval_phase/train_stream/Task000"]



        # ============================
        # Compute per-department losses
        # ============================
        loss_per_dep = [[] for i in experiment_parameters["dept_ids"]]
        for i in range(exp_id+1):
            exp_i = benchmark.train_stream[i]
            main_test_ds_i = copy.copy(exp_i.dataset)
            for itr_dep, dept_id in enumerate(experiment_parameters["dept_ids"]):
                dept_indices = torch.where(main_test_ds_i[:][3] == dept_id)
                subexp_ds = AvalancheSubset(main_test_ds_i, indices=dept_indices[0])
                if len(subexp_ds) > 0:
                    exp_i.dataset = subexp_ds
                    res = strategy.eval(exp_i)
                    loss_dept_i = res[f"Loss_Exp/eval_phase/train_stream/Task000/Exp{i:03d}"]
                else:
                    loss_dept_i = None
                loss_per_dep[itr_dep].append(loss_dept_i)

        if log_wandb:
            wandb.log({"experience/loss_train": loss_train_exp}, step=global_iter)
            wandb.log({"experience/loss_exp": loss_eval_exp}, step=global_iter)
            wandb.log({"experience/loss_exp_allseen": loss_eval_exp_allseen}, step=global_iter)

            for (itr_dep, dept_id) in enumerate(experiment_parameters["dept_ids"]):
                dep_losses = loss_per_dep[itr_dep]
                if dep_losses[-1] != None:
                    wandb.log({f"dept/loss_dept{dept_id}": dep_losses[-1]}, step=global_iter)
                dep_losses = [l for l in dep_losses if l != None]
                if len(dep_losses) > 0:
                    wandb.log({f"dept_avg/loss_dept_avg{dept_id}": np.mean(dep_losses)}, step=global_iter)
            # save checkpoint
            torch.save(strategy.model.state_dict(), os.path.join(output_path, f"ckpt_{run_name}_{exp_id}.pt"))

        
        # increment global iterator
        global_iter += 1

    # ============================
    # Compute FPs and FNs in the Final Experience
    # ============================

    last_exp_id = len(benchmark.train_stream) - 1
    rec, info_rec = mha.compute_rec_ratio(strategy, benchmark.train_stream[last_exp_id].dataset,experiment_parameters)
    prec, info_prec = mha.compute_prec_ratio(strategy, benchmark.train_stream[last_exp_id].dataset,experiment_parameters)


    
    print("rec:",rec)
    print("prec:",prec)

    if log_wandb:
        # Log scalar recall
        wandb.log({"rec": rec}, step=global_iter)

        # Create Recall Table: each row contains [department, rec_loss]
        rec_data = list(zip(info_rec["depts"], info_rec["rec_losses"]))
        rec_table = wandb.Table(columns=["Dept", "Reconstruction Loss"], data=rec_data)
        wandb.log({"Recall Table": rec_table}, step=global_iter)

        # Log scalar precision
        wandb.log({"prec": prec}, step=global_iter)

        # Create Precision Table: each row contains [department, rec_loss]
        prec_data = list(zip(info_prec["depts"], info_prec["rec_losses"]))
        prec_table = wandb.Table(columns=["Dept", "Reconstruction Loss"], data=prec_data)
        wandb.log({"Precision Table": prec_table}, step=global_iter)

        wandb.finish()