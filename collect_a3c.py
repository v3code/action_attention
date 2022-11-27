import torch

# noinspection PyUnresolvedReferences
import action_attention.envs
from action_attention import utils, paths
from action_attention.stack import Stack
from action_attention.stacks.dataset.collect import InitEnvAndSeed, CollectA3CAtariAndSave
from action_attention.stacks.model.a3c import InitModel

ex = utils.setup_experiment("config/a3c_pong.json")
ex.add_config(paths.CFG_MODEL_PONG_A3C)


@ex.capture()
def get_model_config(model_config):
    return model_config

@ex.config
def config():
    config_path='config/a3c_pong.json'
    save_path = None
    num_episodes = None
    load_path = None
    env_id = None
    seed = None
    num_steps = 50
    crop = None
    min_burnin = 58
    max_burnin = 100
    dedup_paths = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'



@ex.automain
def main(save_path, device, num_episodes, min_burnin, dedup_paths, max_burnin, env_id, seed, crop, num_steps, load_path):

    logger = utils.Logger()
    model_config = get_model_config()
    stack = Stack(logger)

    stack.register(InitEnvAndSeed(
        env_id=env_id,
        seed=seed
    ))
    stack.register(InitModel(
        model_config=model_config,
        learning_rate=5e-4,
        load_path=load_path,
        device=device
    ))
    stack.register(CollectA3CAtariAndSave(
        save_path=save_path,
        num_episodes=num_episodes,
        num_steps=num_steps,
        max_burnin=max_burnin,
        min_burnin=min_burnin,
        crop=crop,
        device=device,
        dedup_paths=dedup_paths
    ))


    stack.forward(None, None)
