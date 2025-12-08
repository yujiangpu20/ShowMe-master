import argparse, os, sys
from tqdm import tqdm
from omegaconf import OmegaConf
from einops import repeat, rearrange
import time

import torch
from pytorch_lightning import seed_everything

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from utils.utils import instantiate_from_config
from torch.utils.data import DataLoader

from utils_infer import (
    VideoInferenceDataset,
    load_model_checkpoint,
    load_lora_checkpoint,
    save_results
)


def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


def get_latent_zero_mask(z):
    b, c, t, h, w = z.shape
    mask = torch.zeros((b, c, t - 1, h, w)).to(z.device)
    return mask


def image_guided_synthesis(
        model,
        prompts,
        videos,
        noise_shape,
        ddim_steps=50,
        ddim_eta=1.0,
        unconditional_guidance_scale=1.0,
        cfg_img=None,
        fs=None,
        text_input=False,
        multiple_cond_cfg=False,
        timestep_spacing='uniform',
        guidance_rescale=0.0,
        mask_frame_cond=False,
        final_frame_prediction=False,
        **kwargs
):
    ddim_sampler = DDIMSampler(model) if not multiple_cond_cfg else DDIMSampler_multicond(model)
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    if not text_input:
        prompts = [""] * batch_size

    # --- Conditioning Setup ---
    cond_frame_index = 0
    img = videos[:, :, cond_frame_index, ...].to(model.device)

    img_emb = model.embedder(img)  # [B, L, D]
    img_emb = model.image_proj_model(img_emb)
    cond_emb = model.get_learned_conditioning(prompts)  # [B, Tq, D]
    cond = {"c_crossattn": [torch.cat([cond_emb, img_emb], dim=1)]}

    z = get_latent_z(model, videos)

    # --- Branching Logic for FFP vs Video ---
    img_cat_cond = None

    if final_frame_prediction:
        if model.model.conditioning_key == 'hybrid':
            img_cat_cond = z[:, :, [cond_frame_index], ...]  # [B, C, 1, H, W]
            cond["c_concat"] = [img_cat_cond]
    else:
        if model.model.conditioning_key == 'hybrid':
            img_cat_cond = z[:, :, 0, :, :].unsqueeze(2)
            if not mask_frame_cond:
                img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
            else:
                mask = get_latent_zero_mask(z)
                img_cat_cond = torch.cat((img_cat_cond, mask), dim=2)

            cond["c_concat"] = [img_cat_cond]

    # --- Unconditional Setup ---
    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            uc_emb = model.get_learned_conditioning(batch_size * [""])
        elif model.uncond_type == "zero_embed":
            uc_emb = torch.zeros_like(cond_emb)

        uc_img = torch.zeros_like(img)  # [B, C, H, W]
        uc_img_emb = model.image_proj_model(model.embedder(uc_img))
        uc = {"c_crossattn": [torch.cat([uc_emb, uc_img_emb], dim=1)]}

        if model.model.conditioning_key == 'hybrid':
            uc["c_concat"] = [img_cat_cond]
    else:
        uc = None

    # Handle cfg_img specific logic
    if multiple_cond_cfg and cfg_img != 1.0:
        uc_2 = {"c_crossattn": [torch.cat([uc_emb, img_emb], dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc_2["c_concat"] = [img_cat_cond]
        kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})

    if final_frame_prediction:
        kwargs.update({"final_frame_prediction": True})

    # --- Sampling ---
    samples, _ = ddim_sampler.sample(
        S=ddim_steps,
        conditioning=cond,
        batch_size=batch_size,
        shape=noise_shape[1:],
        verbose=False,
        unconditional_guidance_scale=unconditional_guidance_scale,
        unconditional_conditioning=uc,
        eta=ddim_eta,
        cfg_img=cfg_img,
        fs=fs,
        timestep_spacing=timestep_spacing,
        guidance_rescale=guidance_rescale,
        **kwargs
    )  # [B, C, T=1, H, W]

    # --- Output Formatting ---
    if final_frame_prediction:
        # FFP Output: [Cond Frame, GT Last Frame, Predicted Frame]
        sample_images = model.decode_first_stage(samples.repeat(1, 1, 3, 1, 1))
        cond_images = model.decode_first_stage(img_cat_cond.repeat(1, 1, 3, 1, 1))
        # Get Ground Truth last frame for comparison
        target_images = model.decode_first_stage(z[:, :, -3:, ...])
        # Concatenate for visual comparison [Start, Target, Pred]
        batch_videos = torch.cat([cond_images, target_images, sample_images], dim=3)
        return batch_videos
    else:
        # Video Output: [Input Sequence + Generated Sequence]
        sample_videos = model.decode_first_stage(samples)
        batch_videos = torch.cat([videos, sample_videos], dim=3)
        return batch_videos


# --------------------------------------------------------------------------------
# Main Execution Loop
# --------------------------------------------------------------------------------
def run_inference(args, gpu_num, gpu_no):
    # Setup Configs
    configs = [OmegaConf.load(cfg) for cfg in args.base]
    config = OmegaConf.merge(*configs)
    model_config = config.pop("model", OmegaConf.create())
    model_config['params']['unet_config']['params']['use_checkpoint'] = False

    # Initialize Model
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    model.perframe_ae = args.perframe_ae

    # Load base checkpoint
    assert os.path.exists(args.base_ckpt), f"Error: base checkpoint {args.base_ckpt} Not Found!"
    model = load_model_checkpoint(model, args.base_ckpt)

    # Load LoRA checkpoint (if provided)
    if args.lora_ckpt is not None:
        assert os.path.exists(args.lora_ckpt), f"Error: LoRA checkpoint {args.lora_ckpt} Not Found!"
        model = load_lora_checkpoint(model, args.lora_ckpt)

    model.eval()

    # Latent Setup
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.out_channels

    # Dirs
    fakedir = os.path.join(args.savedir, "samples")
    os.makedirs(fakedir, exist_ok=True)

    # Dataset
    dataset = VideoInferenceDataset(
        meta_path=args.metadata_dir,
        data_dir=args.data_dir,
        dataset=args.dataset,
        video_length=args.video_length,
        resolution=(args.height, args.width)
    )

    # Dataloader splitting
    num_samples = len(dataset)
    samples_split = num_samples // gpu_num
    start_idx = samples_split * gpu_no
    end_idx = samples_split * (gpu_no + 1) if gpu_no < (gpu_num - 1) else num_samples
    subset_indices = list(range(start_idx, end_idx))

    loader = DataLoader(dataset, batch_size=1, sampler=subset_indices, num_workers=8, pin_memory=True)
    print(f"Starting Inference | Mode: {'Image Manipulation' if args.final_frame_prediction else 'Video Generation'}")

    start_time = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch_idx, batch in enumerate(tqdm(loader, desc='Inference')):
            video_name = batch['video_name'][0]
            prompt = batch['prompt'][0]
            frames = batch['frames'].to(gpu_no)

            repeated_frames = frames.repeat(args.n_samples, 1, 1, 1, 1)
            repeated_prompts = [prompt] * args.n_samples

            # Determine noise shape
            if args.final_frame_prediction:
                repeated_noise_shape = [args.n_samples, channels, 1, h, w]
            else:
                repeated_noise_shape = [args.n_samples, channels, args.video_length, h, w]

            batch_vis = image_guided_synthesis(
                model,
                repeated_prompts,
                repeated_frames,
                repeated_noise_shape,
                ddim_steps=args.ddim_steps,
                ddim_eta=args.ddim_eta,
                unconditional_guidance_scale=args.unconditional_guidance_scale,
                cfg_img=args.cfg_img,
                fs=args.frame_stride,
                text_input=args.text_input,
                multiple_cond_cfg=args.multiple_cond_cfg,
                timestep_spacing=args.timestep_spacing,
                guidance_rescale=args.guidance_rescale,
                final_frame_prediction=args.final_frame_prediction,
                mask_frame_cond=args.mask_frame_cond,
            )

            save_results(
                batch_vis,
                video_name,
                fakedir,
                fps=1 if args.final_frame_prediction else args.video_length // 2,
                # save_as_image=args.final_frame_prediction  # Only save as JPG if FFP
            )

    print(f"Saved in {args.savedir}. Time used: {(time.time() - start_time):.2f} seconds")


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_ckpt", type=str, required=True,
                        help="path to base DynamiCrafter checkpoint (full model weights)")
    parser.add_argument("--lora_ckpt", type=str, default=None,
                        help="LoRA checkpoint path (optional)")
    parser.add_argument("--base", "-b", nargs="*",
                        default=[], help="paths to base configs")
    # --- Mode Switch ---
    parser.add_argument("--final_frame_prediction", action='store_true', default=False,
                        help="If True, acts as Image editor. If False, acts as Video predictor.")

    parser.add_argument("--dataset", type=str, required=True, choices=['ssv2', 'epic100', 'ego4d'])
    parser.add_argument("--metadata_dir", type=str, required=True, help="metadata path (jsonl)")
    parser.add_argument("--data_dir", type=str, required=True, help="video data path")
    parser.add_argument("--savedir", type=str, required=True, help="results saving path")
    parser.add_argument("--video_length", type=int, default=16, help="12 for ssv2 and 16 for epic/ego4d")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt")
    parser.add_argument("--bs", type=int, default=1, help="batch size")
    parser.add_argument("--seed", type=int, default=20230211, help="seed from DynamiCrafter")

    # Model Params
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=1.0)
    parser.add_argument("--mask_frame_cond", action='store_true', default=False)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=7.5)
    parser.add_argument("--cfg_img", type=float, default=None)
    parser.add_argument("--frame_stride", type=int, default=3, help="range recommended: 1-5")
    parser.add_argument("--timestep_spacing", type=str, default="uniform")
    parser.add_argument("--guidance_rescale", type=float, default=0.0)
    parser.add_argument("--text_input", action='store_false', default=True)
    parser.add_argument("--perframe_ae", action='store_true', default=False)
    parser.add_argument("--multiple_cond_cfg", action='store_true', default=False)

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args, unknown = parser.parse_known_args()

    rank, gpu_num = 0, 1
    seed_everything(args.seed + rank)
    run_inference(args, gpu_num, rank)