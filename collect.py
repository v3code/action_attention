# noinspection PyUnresolvedReferences
import action_attention.envs
from action_attention import utils
from action_attention.stack import Stack
from action_attention.stacks.dataset.collect import InitEnvAndSeed, CollectRandomAndSave, CollectRandomAtariAndSave

ex = utils.setup_experiment("collect")


@ex.config
def config():

    save_path = None
    num_episodes = None
    env_id = None
    seed = None
    atari = False
    num_steps = 50
    warmstart = None
    crop = None



@ex.automain
def main(save_path, num_episodes, env_id, atari, seed, warmstart, crop, num_steps):

    logger = utils.Logger()
    stack = Stack(logger)

    stack.register(InitEnvAndSeed(
        env_id=env_id,
        seed=seed
    ))

    if atari:
        stack.register(CollectRandomAtariAndSave(
            save_path=save_path,
            num_episodes=num_episodes,
            num_steps=num_steps,
            warmstart=warmstart,
            crop=crop,

        ))
    else:
        stack.register(CollectRandomAndSave(
            save_path=save_path,
            num_episodes=num_episodes
        ))


    stack.forward(None, None)
