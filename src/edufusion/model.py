import numpy as np
import torch
import torchvision.transforms.functional as tvF
from tqdm import tqdm

from .scheduler import Scheduler
from .utils import load_models


class StableDiffusion:
    def __init__(self, device: str = "cuda"):
        self.device = device

        self.scheduler = Scheduler()
        self.vae, self.text_encoder, self.unet = load_models(device=device)
        self.latent_scale_factor = 0.18215

    @torch.no_grad()
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
    ):
        with torch.autocast(device_type=self.device, dtype=torch.float16):
            schedule = Scheduler()

            # Text condition
            context = self.text_encoder.encode([negative_prompt, prompt])

            # Starting latent
            if zT is not None:
                zt = zT
            else:
                random_number_generator = torch.Generator(device=self.device)
                random_number_generator.manual_seed(seed)
                zt = torch.randn((1, 4, 64, 64), generator=random_number_generator, device=self.device)

            if start_step is None:
                start_step = schedule.n_timesteps

            timesteps = torch.linspace(start_step-1, 0, num_steps+1, dtype=torch.int32, device=self.device)
            pbar = tqdm(range(num_steps)) if verbose else range(num_steps)

            # Save history
            pred_x0_history = []
            pred_xt_history = []
            if save_x0_history:
                pred_x0_viz = tvF.to_pil_image((self.vae.decode(zt / self.latent_scale_factor)[0] / 2 + 0.5).clip(0, 1))
                pred_x0_history.append(pred_x0_viz)
            if save_xt_history:
                pred_xt_viz = tvF.to_pil_image((self.vae.decode(zt / self.latent_scale_factor)[0] / 2 + 0.5).clip(0, 1))
                pred_xt_history.append(pred_xt_viz)

            for i in pbar:
                t = timesteps[i]
                t_next = timesteps[i + 1]

                # Noise prediction
                eps_uncond, eps = self.unet(
                    torch.cat([zt, zt]),
                    timesteps=torch.tensor([t, t], device=self.device),
                    context=context
                ).chunk(2)

                # Classifier-free guidance
                eps = eps_uncond + guidance_scale * (eps - eps_uncond)

                # Predicted x_0
                pred_z0 = (zt - schedule.sqrt_one_minus_alphas[t] * eps) / schedule.sqrt_alphas[t]

                # Save x_0 history
                if save_x0_history:
                    pred_x0_viz = tvF.to_pil_image((self.vae.decode(pred_z0 / self.latent_scale_factor)[0] / 2 + 0.5).clip(0, 1))
                    pred_x0_history.append(pred_x0_viz)

                # DDIM step
                if t_next == -1:
                    zt = pred_z0
                else:
                    sigma_t = eta * schedule.sqrt_one_minus_alphas[t_next] / schedule.sqrt_one_minus_alphas[t] * torch.sqrt(1 - schedule.alphas[t] / schedule.alphas[t_next])
                    dir_xt = torch.sqrt(1 - schedule.alphas[t_next] - sigma_t ** 2) * eps
                    zt = schedule.sqrt_alphas[t_next] * pred_z0 + dir_xt + sigma_t * torch.randn_like(zt)

                # Save x_t history
                if save_xt_history:
                    pred_xt_viz = tvF.to_pil_image((self.vae.decode(zt / self.latent_scale_factor)[0] / 2 + 0.5).clip(0, 1))
                    pred_xt_history.append(pred_xt_viz)

            # Decode
            output_image = self.vae.decode(zt / self.latent_scale_factor)
            output_image = tvF.to_pil_image((output_image[0] / 2 + 0.5).clip(0, 1))
        
        if save_x0_history and save_xt_history:
            return output_image, pred_x0_history, pred_xt_history
        if save_x0_history:
            return output_image, pred_x0_history
        if save_xt_history:
            return output_image, pred_xt_history
        return output_image
