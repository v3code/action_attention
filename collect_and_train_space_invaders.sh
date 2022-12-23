#!/bin/bash
if [ ! -f ./model/spinv/model.80.tar ]
then
  echo "No model file, downloading..."
  wget https://github.com/greydanus/baby-a3c/raw/master/spaceinvaders-v4/model.80.tar -P ./model/spinv/
fi
python collect_a3c.py with "seed=1" "min_burnin=50" "max_burnin=300" "crop=30, 200" "num_steps=10" "env_id=SpaceInvadersDeterministic-v4" "load_path=model/spinv/" "save_path=data/spinv_train" "num_episodes=10000"
python collect_a3c.py with "seed=2" "min_burnin=50" "max_burnin=300" "crop=30, 200" "num_steps=10" "env_id=SpaceInvadersDeterministic-v4" "dedup_paths=data/spinv_train" "load_path=model/spinv/" "save_path=data/spinv_eval" "num_episodes=1000"
python run_cswm_pong.py with "dataset_path=data/spinv_train" "model_config.copy_action=True" "eval_dataset_path=data/spinv_eval" "seed=44" "epochs=100" "eval_steps=1,5,10" "learning_rate=5e-4" "model_save_path=data/cswm_spinv.pt"