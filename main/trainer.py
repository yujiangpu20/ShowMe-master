import argparse, os, sys, datetime
from omegaconf import OmegaConf
from transformers import logging as transf_logging
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
import torch
# torch.set_float32_matmul_precision('medium')  # for NVIDIA RTX 6000 Ada Generation

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.utils import instantiate_from_config
from utils_train import get_trainer_callbacks, get_trainer_logger, get_trainer_strategy
from utils_train import set_logger, init_workspace, load_checkpoints


class RewardModels:
    """
    A simple container that holds external reward models *outside* of your main LightningModule.
    """
    def __init__(self, load_edge=False, load_depth=False, load_motion=False):
        self.hedge_model = None
        self.ds_model = None
        self.raft_model = None

        if load_edge:
            self._init_hedge_model()
            logger.info("Loaded HED edge detection model...")
        if load_depth:
            self._init_depth_model()
            logger.info("Loaded MiDaS depth estimation model...")
        if load_motion:
            self._init_motion_model()
            logger.info("Loaded RAFT optical flow model...")

    def _init_hedge_model(self):
        from lvdm.models.hed_edge import HedEdge
        self.hedge_model = HedEdge()
        state_dict = torch.hub.load_state_dict_from_url(
            url='http://content.sniklaus.com/github/pytorch-hed/network-bsds500.pytorch',
            file_name='hed-bsds500'
        )
        self.hedge_model.load_state_dict({
            strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in state_dict.items()
        })
        self.hedge_model.eval()
        for param in self.hedge_model.parameters():
            param.requires_grad = False

    def _init_depth_model(self):
        self.ds_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.ds_model.eval()
        for param in self.ds_model.parameters():
            param.requires_grad = False

    def _init_motion_model(self):
        # Example for optical flow (RAFT):
        from torchvision.models.optical_flow import raft_large
        self.raft_model = raft_large(pretrained=True)
        self.raft_model.eval()
        for param in self.raft_model.parameters():
            param.requires_grad = False

    def to(self, device):
        """Manually move these models to a particular device, if needed."""
        if self.hedge_model is not None:
            self.hedge_model.to(device)
        if self.ds_model is not None:
            self.ds_model.to(device)
        if self.raft_model is not None:
            self.raft_model.to(device)

    def eval(self):
        """Ensure eval-mode. (Already set, but here's a helper if you need it.)"""
        if self.hedge_model is not None:
            self.hedge_model.eval()
        if self.ds_model is not None:
            self.ds_model.eval()
        if self.raft_model is not None:
            self.raft_model.eval()


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--projectname", type=str, default="image2video")
    parser.add_argument("--seed", "-s", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--name", "-n", type=str, default="", help="experiment name, as saving folder")

    parser.add_argument("--base", "-b", nargs="*", metavar="base_config.yaml", help="paths to base configs. Loaded from left-to-right. "
                            "Parameters can be overwritten or added with command-line options of the form `--key value`.", default=list())
    
    parser.add_argument("--train", "-t", action='store_true', default=False, help='train')
    parser.add_argument("--val", "-v", action='store_true', default=False, help='val')
    parser.add_argument("--test", action='store_true', default=False, help='test')

    parser.add_argument("--wandb", type=bool, nargs="?", default=False, help="log to wandb")
    parser.add_argument("--logdir", "-l", type=str, default="logs", help="directory for logging dat shit")
    parser.add_argument("--auto_resume", action='store_true', default=False, help="resume from full-info checkpoint")
    parser.add_argument("--auto_resume_weight_only", action='store_true', default=False, help="resume from weight-only checkpoint")
    parser.add_argument("--debug", "-d", action='store_true', default=False, help="enable post-mortem debugging")

    return parser
    
def get_nondefault_trainer_args(args):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    default_trainer_args = parser.parse_args([])
    return sorted(k for k in vars(default_trainer_args) if getattr(args, k) != getattr(default_trainer_args, k))


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    local_rank = int(os.environ.get('LOCAL_RANK'))
    global_rank = int(os.environ.get('RANK'))
    num_rank = int(os.environ.get('WORLD_SIZE'))

    parser = get_parser()
    ## Extends existing argparse by default Trainer attributes
    parser = Trainer.add_argparse_args(parser)
    args, unknown = parser.parse_known_args()
    ## disable transformer warning
    transf_logging.set_verbosity_error()
    seed_everything(args.seed)

    ## yaml configs: "model" | "data" | "lightning"
    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create()) 

    ## setup workspace directories
    workdir, ckptdir, cfgdir, loginfo = init_workspace(args.name, args.logdir, config, lightning_config, global_rank)
    logger = set_logger(logfile=os.path.join(loginfo, 'log_%d:%s.txt'%(global_rank, now)))
    logger.info("@lightning version: %s [>=1.8 required]"%(pl.__version__))  

    ## MODEL CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configing Model *****")
    config.model.params.logdir = workdir
    model = instantiate_from_config(config.model)

    ## load checkpoints
    model = load_checkpoints(model, config.model)

    ## Decide whether to load reward models
    reward_tuning = config.model.params.get("reward_tuning", False)
    reward_type = config.model.params.get("reward_type", [])
    if reward_tuning:
        reward_models = RewardModels(
            load_edge=("edge" in reward_type),
            load_depth=("depth" in reward_type),
            load_motion=("motion" in reward_type),
        )
        model.reward_models = reward_models
    else:
        model.reward_models = None

    ## register_schedule again to make ZTSNR work
    if model.rescale_betas_zero_snr:
        model.register_schedule(given_betas=model.given_betas, beta_schedule=model.beta_schedule, timesteps=model.timesteps,
                                linear_start=model.linear_start, linear_end=model.linear_end, cosine_s=model.cosine_s)

    ## update trainer config
    for k in get_nondefault_trainer_args(args):
        trainer_config[k] = getattr(args, k)
        
    num_nodes = trainer_config.num_nodes
    ngpu_per_node = trainer_config.devices
    logger.info(f"Running on {num_rank}={num_nodes}x{ngpu_per_node} GPUs")

    ## setup learning rate
    base_lr = config.model.base_learning_rate
    bs = config.data.params.batch_size
    if getattr(config.model, 'scale_lr', True):
        model.learning_rate = num_rank * bs * base_lr
    else:
        model.learning_rate = base_lr


    ## DATA CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configing Data *****")
    data = instantiate_from_config(config.data)
    data.setup()
    for k in data.datasets:
        logger.info(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")


    ## TRAINER CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configing Trainer *****")
    if "accelerator" not in trainer_config:
        trainer_config["accelerator"] = "gpu"

    ## setup trainer args: pl-logger and callbacks
    trainer_kwargs = dict()
    trainer_kwargs["num_sanity_val_steps"] = 0
    logger_cfg = get_trainer_logger(lightning_config, workdir, args.debug)
    # logger_cfg = get_trainer_logger(now, lightning_config, workdir, args, config)   # wandb logger
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)
    
    ## setup callbacks
    callbacks_cfg = get_trainer_callbacks(lightning_config, config, workdir, ckptdir, logger)
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    strategy_cfg = get_trainer_strategy(lightning_config)
    trainer_kwargs["strategy"] = strategy_cfg if type(strategy_cfg) == str else instantiate_from_config(strategy_cfg)
    trainer_kwargs['precision'] = lightning_config.get('precision', 32)
    trainer_kwargs["sync_batchnorm"] = False

    trainer_args = argparse.Namespace(**trainer_config)
    trainer = Trainer.from_argparse_args(trainer_args, **trainer_kwargs)

    ## Running LOOP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Running the Loop *****")
    if args.train:
        try:
            if "strategy" in lightning_config and lightning_config['strategy'].startswith('deepspeed'):
                logger.info("<Training in DeepSpeed Mode>")
                ## deepspeed
                if trainer_kwargs['precision'] == 16:
                    with torch.cuda.amp.autocast():
                        trainer.fit(model, data)
                else:
                    trainer.fit(model, data)
            else:
                logger.info("<Training in DDPSharded Mode>") ## this is default
                ## ddpsharded
                trainer.fit(model, data)
        except Exception:
            #melk()
            raise

    # if args.val:
    #     trainer.validate(model, data)
    # if args.test or not trainer.interrupted:
    #     trainer.test(model, data)