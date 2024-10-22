# Edufusion: Stable Diffusion 1.5 for Educational Purposes

<img src="./.github/edufusion.png" >

Edufusion is a DIY implementation of Stable Diffusion 1.5 with minimal dependencies, designed for educational purposes.

This repository provides only the core components of Stable Diffusion 1.5 (VAE, CLIP Tokenizer, CLIP Text Encoder and UNet) along with their corresponding weights. The implementation of the sampling loop is left as an exercise for learners.

The goals of this project are:

- Foster a deep understanding of Diffusion Models by challenging users to implement sampling methods using the provided pre-trained LDM components.
- Provide a minimal implementation of LDM, enabling learners to grasp the functionality of each component without navigating complex dependency trees.
- Enable users to easily conduct hands-on experimentation with diffusion models to deepen their understanding and to potentially contribute new innovations to the field.


## Sample usage

The package can be installed like this:
```bash
git clone https://github.com/AndranikSargsyan/Edufusion
cd Edufusion
pip install -e .
```

Then download the model weights by executing the following commands:
```bash
mkdir -p models
wget -ncv --show-progress -O models/text-encoder-sd-v1-5-fp16.pt https://huggingface.co/andraniksargsyan/stable-diffusion-v1-5/resolve/main/text-encoder-sd-v1-5-fp16.pt?download=true
wget -ncv --show-progress -O models/vae-sd-v1-5-fp16.pt https://huggingface.co/andraniksargsyan/stable-diffusion-v1-5/resolve/main/vae-sd-v1-5-fp16.pt?download=true
wget -ncv --show-progress -O models/unet-sd-v1-5-fp16.pt https://huggingface.co/andraniksargsyan/stable-diffusion-v1-5/resolve/main/unet-sd-v1-5-fp16.pt?download=true
```

After the installation the components can be used as demonstrated in the following sections.

### Variational AutoEncoder (VAE)
In Stable Diffusion, the VAE helps in encoding high-dimensional images into a lower-dimensional latent space, reducing the complexity for UNet processing.

```python
import torch
import torchvision.transforms.functional as tvF
from PIL import Image

from edufusion import AutoencoderKL

vae = AutoencoderKL().to(device="cuda").half()
vae.load_state_dict(torch.load("path/to/vae-sd-v1-5-fp16.pt", weights_only=True))
img = Image.open("path/to/image.jpg").resize((512, 512))
img_tensor = 2.0 * tvF.pil_to_tensor(img).unsqueeze(0) / 255 - 1.0
img_tensor = img_tensor.to("cuda").half()
with torch.no_grad():
    z0 = vae.encode(img_tensor).mean
    x = vae.decode(z0)
reconstructed_img = tvF.to_pil_image((x[0] / 2 + 0.5).clip(0, 1))
```

### CLIP Text Encoder
The CLIP Text Encoder in Stable Diffusion converts text prompts into vector representations, which guide the image generation process.

```python
from edufusion import FrozenCLIPEmbedder
text_encoder = FrozenCLIPEmbedder()
text_encoder = text_encoder.to(device="cuda").half()
text_encoder.load_state_dict(torch.load("path/to/text-encoder-sd-v1-5-fp16.pt", weights_only=True))
encoded_text = text_encoder.encode(['The quick brown fox jumps over the lazy dog'])
print(encoded_text.shape)
```

### UNet

The UNet is the denoiser network in DDPM theory.

```python
from edufusion import UNetModel
unet = UNetModel().to(device="cuda").half()
unet.load_state_dict(torch.load("./models/unet-sd-v1-5-fp16.pt", weights_only=True))
print("Total number of UNet parameters:", sum(p.numel() for p in unet.parameters()))
```

### Tasks to DIY
- Verify the reconstruction ability of the provided VAE,
- Implement [DDPM](https://arxiv.org/abs/2006.11239) reverse process with classifier-free guidance,
- Implement [DDIM](https://arxiv.org/abs/2010.02502) reverse process with classifier-free guidance,
- Implement deterministic DDIM forward process and perform DDIM inversion to verify the reconstruction quality,
- Perform [SDEdit](https://arxiv.org/abs/2108.01073)-like image editing,
- Implement [Blended Latent Diffusion](https://arxiv.org/abs/2206.02779) inpainting method,
- ...


## Acknowledgements
The code in this repository is based on
[stablediffusion](https://github.com/Stability-AI/stablediffusion), [CLIP](https://github.com/openai/CLIP) and [🤗 Transformers](https://github.com/huggingface/transformers).