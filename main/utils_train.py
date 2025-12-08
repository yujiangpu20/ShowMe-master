import os, re
from omegaconf import OmegaConf
import logging
mainlogger = logging.getLogger('mainlogger')

import torch
from collections import OrderedDict


def init_workspace(name, logdir, model_config, lightning_config, rank=0):
    workdir = os.path.join(logdir, name)
    ckptdir = os.path.join(workdir, "checkpoints")
    cfgdir = os.path.join(workdir, "configs")
    loginfo = os.path.join(workdir, "loginfo")

    # Create logdirs and save configs (all ranks will do to avoid missing directory error if rank:0 is slower)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)
    os.makedirs(loginfo, exist_ok=True)

    if rank == 0:
        if "callbacks" in lightning_config and 'metrics_over_trainsteps_checkpoint' in lightning_config.callbacks:
            os.makedirs(os.path.join(ckptdir, 'trainstep_checkpoints'), exist_ok=True)
        OmegaConf.save(model_config, os.path.join(cfgdir, "model.yaml"))
        OmegaConf.save(OmegaConf.create({"lightning": lightning_config}), os.path.join(cfgdir, "lightning.yaml"))
    return workdir, ckptdir, cfgdir, loginfo


def check_config_attribute(config, name):
    if name in config:
        value = getattr(config, name)
        return value
    else:
        return None


def get_trainer_callbacks(lightning_config, config, logdir, ckptdir, logger):
    default_callbacks_cfg = {
        "batch_logger": {
            "target": "callbacks.ImageLogger",
            "params": {
                "save_dir": logdir,
                "batch_frequency": 1000,
                "max_images": 4,
                "clamp": True,
            }
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                "log_momentum": False
            }
        },
        "cuda_callback": {
            "target": "callbacks.CUDACallback"
        },
    }

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    return callbacks_cfg


def get_trainer_logger(lightning_config, logdir, on_debug):
    default_logger_cfgs = {
        "tensorboard": {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "save_dir": logdir,
                "name": "tensorboard",
            }
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.CSVLogger",
            "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
    }
    os.makedirs(os.path.join(logdir, "tensorboard"), exist_ok=True)
    default_logger_cfg = default_logger_cfgs["tensorboard"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    return logger_cfg


def get_trainer_strategy(lightning_config):
    default_strategy_dict = {
        "target": "pytorch_lightning.strategies.DDPShardedStrategy"
    }
    if "strategy" in lightning_config:
        strategy_cfg = lightning_config.strategy
        return strategy_cfg
    else:
        strategy_cfg = OmegaConf.create()

    strategy_cfg = OmegaConf.merge(default_strategy_dict, strategy_cfg)
    return strategy_cfg


def load_checkpoints(model, model_cfg):
    def rename_keys_for_256x256_model(state_dict):
        new_pl_sd = OrderedDict()
        for k, v in state_dict.items():
            if "framestride_embed" in k:
                new_key = k.replace("framestride_embed", "fps_embedding")
                new_pl_sd[new_key] = v
            else:
                new_pl_sd[k] = v
        return new_pl_sd

    if check_config_attribute(model_cfg, "pretrained_checkpoint"):
        pretrained_ckpt = model_cfg.pretrained_checkpoint
        assert os.path.exists(pretrained_ckpt), "Error: Pre-trained checkpoint NOT found at:%s" % pretrained_ckpt
        mainlogger.info(">>> Loading weights from pretrained checkpoint...")

        pl_sd = torch.load(pretrained_ckpt, map_location="cpu")
        if 'state_dict' in pl_sd.keys():
            pl_sd = pl_sd["state_dict"]
        else:
            # deepspeed
            new_pl_sd = OrderedDict()
            for key in pl_sd['module'].keys():
                new_pl_sd[key[16:]] = pl_sd['module'][key]
            pl_sd = new_pl_sd

        if '256' in pretrained_ckpt:
            pl_sd = rename_keys_for_256x256_model(pl_sd)

        # Check for missing and dimension mismatched keys
        model_state_dict = model.state_dict()
        missing_keys = set(model_state_dict.keys()) - set(pl_sd.keys())
        mismatched_keys = []

        for key in model_state_dict.keys():
            if key in pl_sd and model_state_dict[key].shape != pl_sd[key].shape:
                mismatched_keys.append((key, model_state_dict[key].shape, pl_sd[key].shape))

        # Print missing and mismatched keys
        if missing_keys:
            mainlogger.warning(">>> Missing keys in the checkpoint: %s" % missing_keys)
        if mismatched_keys:
            mainlogger.error(">>> Mismatched keys found in checkpoint:")
            for key, model_shape, ckpt_shape in mismatched_keys:
                mainlogger.error(f"Key: {key}, Model shape: {model_shape}, Checkpoint shape: {ckpt_shape}")
            raise RuntimeError("Error: Mismatched keys found in checkpoint")

        if not missing_keys:
            model.load_state_dict(pl_sd, strict=True)
            mainlogger.info(">>> Loaded weights from pretrained checkpoint: %s" % pretrained_ckpt)
        else:
            # Load state dict with strict=False to handle missing keys
            model.load_state_dict(pl_sd, strict=False)
            mainlogger.info(">>> Loaded weights from pretrained checkpoint with missing keys: %s" % pretrained_ckpt)
    else:
        mainlogger.info(">>> Start training from scratch")

    # 2) Load STAGE-1 LoRA checkpoint on top (if provided)
    if check_config_attribute(model_cfg, "pretrained_lora_checkpoint"):
        lora_ckpt = model_cfg.pretrained_lora_checkpoint
        assert os.path.exists(lora_ckpt), \
            "Error: LoRA checkpoint NOT found at:%s" % lora_ckpt
        mainlogger.info(f">>> Loading Stage 1 LoRA weights from {lora_ckpt} ...")

        # Load raw file
        ckpt = torch.load(lora_ckpt, map_location="cpu")

        # Handle 'lora_state_dict' wrapper key from tools.py
        if "lora_state_dict" in ckpt:
            lora_sd = ckpt["lora_state_dict"]
        elif "state_dict" in ckpt:
            lora_sd = ckpt["state_dict"]
        else:
            lora_sd = ckpt

        # Load into the TOP-LEVEL model (DiffusionWrapper)
        # using strict=False allows loading purely the LoRA keys without needing the base keys present in this file.
        missing, unexpected = model.load_state_dict(lora_sd, strict=False)

        mainlogger.info(f">>> Stage 1 LoRA Loaded. Unexpected keys: {len(unexpected)} (Should be 0).")
        if len(unexpected) > 0:
            mainlogger.warning(f"Unexpected keys in LoRA file: {unexpected}")

    return model


def set_logger(logfile, name='mainlogger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s-%(levelname)s: %(message)s"))
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger