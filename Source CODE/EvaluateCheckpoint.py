import os

# limit the number of threads
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6
print("NUMBER OF THREADS ARE LIMITED NOW ...")


# import python libraries
import argparse
import datetime as dt
import numpy as np
import time
import torch

# import project libraries
import UtilsHandler.UtilsHandler as UtilsHandler



def evaluate_continual_checkpoint(experiment_parameters):
    # Initialize handlers
    uha = UtilsHandler()
    sha = StrategyHandler()
    bha = BenchmarkHandler()
    mha = MetricHandler()

    # Set device
    device_str = experiment_parameters.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    experiment_parameters['device'] = device

    # Set seed
    seed = experiment_parameters.get('seed', 1234)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load dataset
    payment_ds = PaymentDataset(experiment_parameters['data_dir'])

    # Build benchmark from config
    perc_matrix = bha.create_percnt_matrix(experiment_parameters)
    exp_assignments, samples_matrix = bha.get_exp_assignment(experiment_parameters, payment_ds, perc_matrix)
    benchmark = bha.get_benchmark(experiment_parameters, payment_ds, exp_assignments)

    # Load strategy and model
    strategy = sha.get_strategy(experiment_parameters, payment_ds)

    # === Load checkpoint ===
    checkpoint_path = experiment_parameters.get('checkpoint_path')
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model weights
        strategy.model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint and strategy.optimizer is not None:
            strategy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and hasattr(strategy, 'scheduler') and strategy.scheduler is not None:
            strategy.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # === Evaluate on last experience ===
    last_exp_id = len(benchmark.train_stream) - 1
    last_exp_dataset = benchmark.train_stream[last_exp_id].dataset

    print(f"Evaluating on last experience (ID: {last_exp_id})...")

    rec, info_rec = mha.compute_rec_ratio(strategy, last_exp_dataset, experiment_parameters)
    prec, info_prec = mha.compute_prec_ratio(strategy, last_exp_dataset, experiment_parameters)

    print("Recall:", rec)
    print("Precision:", prec)

    return {
        "recall": rec,
        "recall_info": info_rec,
        "precision": prec,
        "precision_info": info_prec,
    }
    return 0

# define main function
if _name_ == "_main_":

    parser = argparse.ArgumentParser(description='deepNadim Experiments')

    # experiment parameter
    parser.add_argument('--seed', help='', nargs='?', type=int, default=1234)

    # dataset and data loading parameter
    parser.add_argument('--dataset', help='', nargs='?', type=str, default='philadelphia', required=True)  # chicago, philadelphia
    parser.add_argument('--data_dir', help='', nargs='?', type=str,
                        default='./100_datasets/philadelphia')  # chicago, philadelphia
    parser.add_argument('--no_workers', help='', nargs='?', type=int, default=0)

    # encoder architecture parameter
    parser.add_argument('--architecture', help='', nargs='?', type=str, default='baseline')
    parser.add_argument('--bottleneck', help='', nargs='?', type=str, default='linear')  ## lrelu, tanh, linear
    parser.add_argument('--architecture_size', help='', nargs='?', type=str, default='small')

    # training parameter
    parser.add_argument('--batch_size', help='', nargs='?', type=int, default=128)
    parser.add_argument('--no_epochs', help='', nargs='?', type=int, default=10)
    parser.add_argument('--no_tasks', help='', nargs='?', type=int, default=6)
    parser.add_argument('--warmup_epochs', help='', nargs='?', type=float, default=100)
    parser.add_argument('--optimizer', help='', nargs='?', type=str, default='adam')
    parser.add_argument('--learning_rate', help='', nargs='?', type=float, default=1e-4)
    parser.add_argument('--eval_epoch', help='', nargs='?', type=int, default=1)
    parser.add_argument('--weight_decay', help='', nargs='?', type=float, default=1e-6)

    # loss parameter
    parser.add_argument('--categorical_loss', help='', nargs='?', type=str, default='bce')  # mse, bce

    # evaluation parameter
    parser.add_argument('--valid_size', help='', nargs='?', type=float, default=1.00)  # 238894
    parser.add_argument('--sample_test', help='', type=str, default='True')
    parser.add_argument('--sample_size', help='', nargs='?', type=int, default=10)

    # number of created artificial anomalies
    parser.add_argument('--global_anomalies', help='', nargs='?', type=int, default=60)  # 60
    parser.add_argument('--local_anomalies', help='', nargs='?', type=int, default=140)  # 140

    # logging parameter
    parser.add_argument('--wandb_logging', help='', type=str, default='True')
    parser.add_argument('--checkpoint_epoch', help='', nargs='?', type=int, default=1)
    parser.add_argument('--checkpoint_save', help='', type=str, default='True')

    # parse script arguments
    parser.add_argument('--strategy', help='', nargs='?', type=str, default='Naive')
    parser.add_argument('--wandb_proj', help='', nargs='?', type=str, default='')
    parser.add_argument('--wandb_entity', help='', nargs='?', type=str, default=None)

    parser.add_argument('--benchmark_configs_path', help='', nargs='?', type=str, default='BenchmarkConfigs/BenchmarkConfigs.yml')
    parser.add_argument('--outputs_path', help='', nargs='?', type=str, default='./outputs')

    # ==========
    # ========== Strategies
    # ==========
    # Training Regime
    parser.add_argument('--training_regime', help='', nargs='?', type=str, default='continual')

    # Replay
    parser.add_argument('--replay_mem_size', help='', nargs='?', type=int, default=500)  # 238894

    # lwf
    parser.add_argument('--lwf_alpha', help='', nargs='?', type=float, default=1.00)  # 238894
    parser.add_argument('--lwf_temperature', help='', nargs='?', type=float, default=1.00)  # 238894

    # ewc
    parser.add_argument('--ewc_lambda', help='', nargs='?', type=float, default=1.00)

    # synaptic intelligence
    parser.add_argument('--si_lambda', help='', nargs='?', type=float, default=1.00)
    parser.add_argument('--si_eps', help='', nargs='?', type=float, default=0.001)

    # parse script arguments
    args = parser.parse_args()
    experiment_parameter = vars(parser.parse_args())

    # determine hardware device
    experiment_parameter['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu").type

    # set deterministic seeds of the experiments
    seed_value = experiment_parameter['seed']
    np.random.seed(seed_value)

    # case: cuda enabled
    if experiment_parameter['device'] == 'cuda:0':
        # set deterministic cuda backends
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # set seeds cuda backend
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)

    # init utilities handler
    uha = UtilsHandler.UtilsHandler()

    # parse boolean args as boolean
    experiment_parameter['checkpoint_save'] = uha.str2bool(experiment_parameter['checkpoint_save'])
    experiment_parameter['wandb_logging'] = uha.str2bool(experiment_parameter['wandb_logging'])

    # parse string args as int
    if args.architecture_size == "small":
        experiment_parameter['encoder_dim'] = [128, 64, 32, 16, 8, 4, 2]
        experiment_parameter['decoder_dim'] = [2, 4, 8, 16, 32, 64, 128]
    elif args.architecture_size == "large":
        experiment_parameter['encoder_dim'] = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2]
        experiment_parameter['decoder_dim'] = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    else:
        raise NotImplementedError

    # load yaml BenchmarkConfigs
    benchmark_configs = uha.load_params(experiment_parameter["benchmark_configs_path"])
    experiment_parameter.update(benchmark_configs)

    # set run name
    run_name = experiment_parameter['training_regime'] + '' + experiment_parameter['strategy'] + '' + \
                experiment_parameter["reduction_type"] + "" + f"{experiment_parameter['dataset']}" + '' + \
                f"s{experiment_parameter['seed']}"
    run_name += '_' + time.strftime("%y-%m-%d-%H-%M-%S")
    experiment_parameter['run_name'] = run_name
    results = evaluate_continual_checkpoint(experiment_parameter)
    print(results)



