#!/bin/bash
if [ ! -f ./model/pong/model.80.tar ]
then
  echo "No model file, downloading..."
  wget https://github.com/greydanus/baby-a3c/raw/master/pong-v4/model.80.tar -P ./model/pong/
fi
python collect_a3c.py with "seed=1" "crop=35, 190" "num_steps=10" "env_id=PongDeterministic-v4" "load_path=model/pong/" "save_path=data/pong_train" "num_episodes=1000"
python collect_a3c.py with "seed=2" "crop=35, 190" "num_steps=10" "env_id=PongDeterministic-v4" "dedup_paths=data/pong_train" "load_path=model/pong/" "save_path=data/pong_eval" "num_episodes=1000"
python run_cswm_pong.py with "dataset_path=data/pong_train" "model_config.copy_action=True" "eval_dataset_path=data/pong_eval" "seed=44" "epochs=100" "eval_steps=1,5,10" "learning_rate=5e-4" "model_save_path=data/cswm_pong.pt"