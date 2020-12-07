import importlib
import os
import re

import megengine as mge


class AverageMeter:
    """computes and stores the average and current value
    """

    def __init__(self, name, fmt=":.3f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def try_load_latest_checkpoint(model, base_dir):
    """try to load the latest checkpoint from `epoch-*-checkpoint.pkl`
    if no such checkpoint exists, the the input model and epoch=0 is returned

    Args:
        model (M.Module): model to load
        base_dir (str): path to directory of checkpoints

    Returns:
        model (M.Module): loaded model
        epoch (int): latest epoch
    """
    # try to find the latest epoch checkpoint
    latest_epoch = -1
    for ckpt_path in os.listdir(base_dir):
        r = re.match(r"epoch-(\d+)-checkpoint\.pkl", ckpt_path)
        if r is not None:
            latest_epoch = max(latest_epoch, int(r.group(1)))

    # load checkpoint
    if latest_epoch != -1:
        checkpoint_path = os.path.join(base_dir, f"epoch-{latest_epoch}-checkpoint.pkl")
        checkpoint_data = mge.load(checkpoint_path)
        epoch = checkpoint_data["epoch"]
        for name, param in model.state_dict().items():
            if name not in checkpoint_data["state_dict"]:
                continue
            param = checkpoint_data["state_dict"][name]

    else:
        epoch = 0
        # mge.module.init.fill_(model.stn.fc1.weight, 0)
        # mge.module.init.fill_(model.stn.fc1.bias, 0)
        # mge.module.init.fill_(model.stn.fc2.weight, 0)
        # mge.module.init.fill_(model.stn.fc2.bias, [1,0,0,0,1,0,0,0,1])
    return model, epoch


def load_config_from_path(config_path):
    spec = importlib.util.spec_from_file_location("configs", config_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.configs
