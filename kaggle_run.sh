#!/bin/bash
python collect.py with "env_id=CubesTrain-v0" "save_path=data/cubes_train" "num_episodes=1000" "seed=42"
python collect.py with "env_id=CubesEval-v0" "save_path=data/cubes_eval" "num_episodes=10000" "seed=42"
python run_cswm.py with "dataset_path=data/cubes_train" "eval_dataset_path=data/cubes_eval" "seed=42" "use_hard_attention=True" "model_config.encoder=large" "epochs=200" "learning_rate=1e-4" "model_save_path=data/models/cswm_ha_cubes.pt"
python run_cswm.py with "dataset_path=data/cubes_train" "eval_dataset_path=data/cubes_eval" "seed=42" "use_soft_attention=True" "model_config.encoder=large" "epochs=200" "learning_rate=1e-4" "model_save_path=data/models/cswm_sa_cubes.pt"