from typing import NamedTuple, List

import PIL.Image
import torch
import torchvision.transforms.functional as tvF
from tqdm.auto import tqdm

from .scheduler import Scheduler
from .utils import load_models


class StableDiffusionOutput(NamedTuple):
    image: PIL.Image.Image
    pred_x0_history: List[PIL.Image.Image]
    pred_xt_history: List[PIL.Image.Image]


class StableDiffusion:
    def __init__(self, device: str = "cuda"):
        self.device = device

        self.scheduler = Scheduler()
        self.vae, self.text_encoder, self.unet = load_models(device=device)
        self.latent_scale_factor = 0.18215

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def sample_ddim(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_steps: int = 50,
        zT: torch.Tensor = None,
        start_step: int = None,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        seed: int = 1,
        verbose: bool = True,
        save_x0_history: bool = False,
        save_xt_history: bool = False
    ) -> StableDiffusionOutput:
        # Starting latent
        if zT is not None:
            zt = zT
        else:
            rng = torch.Generator(device=self.device).manual_seed(seed)
            zt = torch.randn((1, 4, 64, 64), generator=rng, device=self.device)

        # Initialize history saving
        pred_x0_history = []
        pred_xt_history = []
        
        # Save initial x_0 and x_t predictions
        if save_x0_history:
            pred_x0_viz = tvF.to_pil_image((self.vae.decode(zt / self.latent_scale_factor)[0] / 2 + 0.5).clip(0, 1))
            pred_x0_history.append(pred_x0_viz)
        if save_xt_history:
            pred_xt_viz = tvF.to_pil_image((self.vae.decode(zt / self.latent_scale_factor)[0] / 2 + 0.5).clip(0, 1))
            pred_xt_history.append(pred_xt_viz)

        # Text conditions
        context = self.text_encoder.encode([negative_prompt, prompt])

        if start_step is None:
            start_step = self.scheduler.n_timesteps - 1

        timesteps = torch.linspace(start_step, 0, num_steps+1, dtype=torch.int32, device=self.device)
        pbar = tqdm(range(num_steps)) if verbose else range(num_steps)

        for i in pbar:
            t = timesteps[i]
            t_next = timesteps[i + 1]

            # Noise prediction
            eps_uncond, eps_cond = self.unet(
                torch.cat([zt, zt]),
                timesteps=torch.tensor([t, t], device=self.device),
                context=context
            ).chunk(2)

            # Classifier-free guidance
            eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            # Predicted x_0
            pred_z0 = (zt - self.scheduler.sqrt_one_minus_alphas[t] * eps_pred) / self.scheduler.sqrt_alphas[t]

            # DDIM step
            if t_next == -1:
                zt = pred_z0
            else:
                sigma_t = eta * self.scheduler.sqrt_one_minus_alphas[t_next] / self.scheduler.sqrt_one_minus_alphas[t] * torch.sqrt(1 - self.scheduler.alphas[t] / self.scheduler.alphas[t_next])
                dir_xt = torch.sqrt(1 - self.scheduler.alphas[t_next] - sigma_t ** 2) * eps_pred
                zt = self.scheduler.sqrt_alphas[t_next] * pred_z0 + dir_xt + sigma_t * torch.randn_like(zt)

            # Save x_0 history
            if save_x0_history:
                pred_x0_viz = tvF.to_pil_image((self.vae.decode(pred_z0 / self.latent_scale_factor)[0] / 2 + 0.5).clip(0, 1))
                pred_x0_history.append(pred_x0_viz)

            # Save x_t history
            if save_xt_history:
                pred_xt_viz = tvF.to_pil_image((self.vae.decode(zt / self.latent_scale_factor)[0] / 2 + 0.5).clip(0, 1))
                pred_xt_history.append(pred_xt_viz)

        # Decode
        output_image = self.vae.decode(zt / self.latent_scale_factor)
        output_image = tvF.to_pil_image((output_image[0] / 2 + 0.5).clip(0, 1))

        return StableDiffusionOutput(
            image=output_image,
            pred_x0_history=pred_x0_history,
            pred_xt_history=pred_xt_history
        )

