import torch.cuda

# noinspection PyUnresolvedReferences
import action_attention.envs
from action_attention import utils, paths
from action_attention.constants import Constants
from action_attention.stack import Stack, Sieve
from action_attention.stacks.dataset.collect import InitEnvAndSeed
from action_attention.stacks.model.train_cswm import InitModel
from action_attention.stacks.model.visualise import Visualise

ex = utils.setup_experiment("visualise")
ex.add_config(paths.CFG_MODEL_CSWM)


@ex.capture()
def get_model_config(model_config):
    # turn variable names into constants
    d = {}
    utils.process_config_dict(model_config, d)
    return d


@ex.config
def config():
    env_id = None
    save_path = None
    seed = None
    use_hard_attention = False
    use_soft_attention = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_load_path = None
    viz_name = ''


@ex.automain
def main(save_path, device, use_hard_attention, use_soft_attention, model_load_path, viz_name, env_id,
         seed):

    model_config = get_model_config()
    logger = utils.Logger()
    stack = Stack(logger)

    stack.register(InitEnvAndSeed(
        env_id=env_id,
        seed=seed
    ))

    stack.register(InitModel(
        model_config=model_config,
        device=device,
        load_path=model_load_path,
        use_hard_attention=use_hard_attention,
        use_soft_attention=use_soft_attention,
        learning_rate=1e-4  # doesn't matter
    ))

    stack.register(Visualise(
        device=device,
        save_path=save_path,
        viz_name=viz_name,
        seed=seed
    ))

    stack.register(Sieve(
        keys={Constants.MODEL}
    ))

    stack.forward(None, None)
