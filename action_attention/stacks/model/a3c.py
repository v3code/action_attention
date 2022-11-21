from action_attention.constants import Constants
from action_attention.modules.A3CBaby import NNPolicy, SharedAdam
from action_attention.stack import StackElement


class InitModel(StackElement):
    # Initialize A3C policy and optimizer
    def __init__(self, model_config, learning_rate, device, load_path=None):
        super().__init__()
        self.model_config = model_config
        self.learning_rate = learning_rate
        self.device = device
        self.load_path = load_path

        self.OUTPUT_KEYS = {Constants.A3C, Constants.A3C_OPTIM}

    def run(self, bundle: dict, viz=False) -> dict:
        print(self.model_config)
        model = NNPolicy(**self.model_config)

        if self.load_path is not None:
            model.try_load(self.load_path)
            model.eval()
        model.to(self.device)

        optimizer = SharedAdam(model.parameters(), lr=self.learning_rate)
        return {Constants.A3C: model, Constants.A3C_OPTIM: optimizer}
