import torch

from action_attention import utils
from action_attention.stack import Stack, Seeds, SacredLog, Sieve, WandbLogger
from action_attention.stacks.model.train_cswm import InitModel, InitTransitionsLoader, InitPathLoader, Train, Eval, \
    InitTransitionsLoaderAtari, InitPathLoaderAtari
from action_attention.stacks.model.slot_correlation import MeasureSlotCorrelation
from action_attention import paths
from action_attention.constants import Constants

ex = utils.setup_experiment("config/pong.json")
ex.add_config(paths.CFG_MODEL_PONG_CSWM)


@ex.capture()
def get_model_config(model_config):

    # turn variable names into constants
    d = {}
    utils.process_config_dict(model_config, d)
    return d


@ex.config
def config():

    seed = None
    use_hard_attention = False
    use_soft_attention = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learning_rate = 1e-4
    batch_size = 1024
    epochs = 100
    model_save_path = None
    model_load_path = None
    project_name = 'CSWM Action attention'
    run_name = None
    dataset_path = "data/shapes_train"
    eval_dataset_path = "data/shapes_eval"
    viz_names = None
    model_conf_name = None
    eval_steps = '1, 5, 10'
    continue_train = False


@ex.automain
def main(seed,continue_train, run_name, project_name, use_hard_attention, use_soft_attention, device, learning_rate, batch_size, epochs, model_save_path,
         model_load_path, dataset_path, eval_dataset_path, viz_names, eval_steps):

    model_config = get_model_config()
    logger = utils.Logger()
    stack = Stack(logger)

    stack.register(Seeds(
        use_torch=True,
        device=device,
        seed=seed
    ))

    stack.register(WandbLogger(run_name=run_name, project_name=project_name))
    stack.register(InitModel(
        model_config=model_config,
        learning_rate=learning_rate,
        device=device,
        load_path=model_load_path,
        use_hard_attention=use_hard_attention,
        use_soft_attention=use_soft_attention,
        continue_train=continue_train
    ))

    if model_load_path is None or continue_train:
        # train model
        stack.register(InitTransitionsLoaderAtari(
            root_path=dataset_path,
            batch_size=batch_size,
            factored_actions=False
        ))
        stack.register(Train(
            epochs=epochs,
            device=device,
            model_save_path=model_save_path
        ))
        stack.register(SacredLog(
            ex=ex,
            keys=[Constants.LOSSES],
            types=[SacredLog.TYPE_LIST]
        ))

    stack.register(Sieve(
        keys={Constants.MODEL}
    ))

    # evaluate model
    for i in eval_steps:

        stack.register(InitPathLoaderAtari(
            root_path=eval_dataset_path,
            path_length=i,
            batch_size=8,
            factored_actions=False
        ))
        stack.register(Eval(
            device=device,
            batch_size=8,
            num_steps=i,
            dedup=True
        ))
        keys = [*[Constants.HITS.name + "_at_{:d}".format(k) for k in Eval.HITS_AT], Constants.MRR]
        stack.register(SacredLog(
            ex=ex,
            keys=keys,
            types=[SacredLog.TYPE_SCALAR for _ in range(len(keys))],
            prefix="{:d}_step".format(i)
        ))
        stack.register(Sieve(
            keys={Constants.MODEL}
        ))

    # calculate correlation between slots
    stack.register(InitPathLoaderAtari(
        root_path=eval_dataset_path,
        path_length=max(eval_steps),
        batch_size=100,
        factored_actions=False
    ))
    stack.register(MeasureSlotCorrelation(
        device=device
    ))
    stack.register(SacredLog(
        ex=ex,
        keys=[Constants.CORRELATION],
        types=[SacredLog.TYPE_SCALAR]
    ))

    stack.forward(None, viz_names)
