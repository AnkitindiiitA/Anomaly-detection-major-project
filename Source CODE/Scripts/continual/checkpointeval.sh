python EvaluateCheckpoint.py --strategy "SynapticIntelligence" --dataset "philadelphia" \
      --data_dir "./Data/phil.csv" \
      --no_epochs 100 --benchmark_configs_path "./BenchmarkConfigs/linear_target-10.yml" \
      --wandb_proj "DeepContinualAuditing" \
      --bottleneck "tanh" --seed $seed --training_regime 'continual'