import argparse

from edufusion.model import StableDiffusion


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A dog sitting in the park, high quality photo")
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--start-step", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--output-path", type=str, default="output.png")
    return parser.parse_args()


def main():
    args = get_args()

    sd = StableDiffusion()
    
    output_image = sd.sample_ddim(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_steps=args.num_steps,
        start_step=args.start_step,
        guidance_scale=args.guidance_scale,
        eta=args.eta,
        seed=args.seed,
        verbose=args.verbose
    )
    output_image.save("output.png")


if __name__ == "__main__":
    main()