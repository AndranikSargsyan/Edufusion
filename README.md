# Edufusion

<img src="./.github/edufusion.png" >

A minimal Stable Diffusion 1.5 implementation for learning and experimentation.

Clean implementations of SD 1.5 components (VAE, CLIP text encoder, UNet) with DDIM sampling. No massive dependency trees, just the core pieces you need to understand how diffusion models work.


## Installation

```bash
git clone https://github.com/AndranikSargsyan/Edufusion
cd Edufusion
pip install -e .
```

That's it. Models download automatically on first use.

## Quick Start

### CLI
Generate an image from the command line:

```bash
python -m edufusion.cli --prompt "A dog sitting in the park, high quality photo" --num-steps 50 --seed 42
```

Options:
- `--prompt`: Your text prompt
- `--negative-prompt`: What you don't want (default: "")
- `--num-steps`: Number of sampling steps (default: 50)
- `--guidance-scale`: CFG scale (default: 7.5)
- `--seed`: Random seed (default: 1)
- `--output-path`: Where to save (default: "output.png")

### Python API

Basic usage:

```python
from edufusion import StableDiffusion

sd = StableDiffusion()
image = sd.sample_ddim(
    prompt="A dog sitting in the park",
    num_steps=50,
    seed=42
)
image.save("output.png")
```

## Component-Level Usage

Access individual components:

```python
import torch
import torchvision.transforms.functional as tvF
from PIL import Image
from edufusion import StableDiffusion

sd = StableDiffusion()

# VAE - compress images to latent space and back
img = Image.open("image.jpg").resize((512, 512))
img_tensor = 2.0 * tvF.pil_to_tensor(img).unsqueeze(0) / 255 - 1.0
img_tensor = img_tensor.to("cuda").half()

with torch.no_grad():
    z0 = sd.vae.encode(img_tensor).mean
    x = sd.vae.decode(z0)
reconstructed = tvF.to_pil_image((x[0] / 2 + 0.5).clip(0, 1))

# Text Encoder - convert prompts to embeddings
encoded = sd.text_encoder.encode(['A dog in the park'])
print(encoded.shape)  # (1, 77, 768)

# UNet - the denoising network
print(f"UNet params: {sum(p.numel() for p in sd.unet.parameters()):,}")
```

## Credits

Code adapted from [stablediffusion](https://github.com/Stability-AI/stablediffusion), [CLIP](https://github.com/openai/CLIP), and [🤗 Transformers](https://github.com/huggingface/transformers).