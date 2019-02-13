# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import argparse
import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config

def main(seed, output_dir, rounds, dataset):
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    urls = {
        'ffhq': 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ', # karras2019stylegan-ffhq-1024x1024.pkl
        'celebahq': 'https://drive.google.com/uc?id=1MGqJl28pN4t7SAtSrPdSRJSQJqahkzUf', # karras2019stylegan-celebahq-1024x1024.pkl
        'bedrooms': 'https://drive.google.com/uc?id=1MOSKeGF0FJcivpBI7s63V9YHloUTORiF', # karras2019stylegan-bedrooms-256x256.pkl
        'cars': 'https://drive.google.com/uc?id=1MJ6iCfNtMIRicihwRorsM3b7mmtmK9c3', # karras2019stylegan-cars-512x384.pkl
        'cats': 'https://drive.google.com/uc?id=1MQywl0FNt6lHu8E_EUqnRbviagS7fbiJ' # karras2019stylegan-cats-256x256.pkl
    }

    url = urls[dataset]

    with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
        _G, _D, Gs = pickle.load(f)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()

    # Generate image.
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    psis = [i/20 for i in range(-14, 15)]
    rnd = np.random.RandomState(42)
    for r in range(rounds):
        # Pick latent vector.
        new_seed = seed if seed else int(np.abs(np.round(rnd.randn() * 1e9)))

        rnd.seed(new_seed)
        latents = rnd.randn(1, Gs.input_shape[1])
        
        for i, psi in enumerate(psis):
            images = Gs.run(latents, None, truncation_psi=psi, randomize_noise=True, output_transform=fmt)

            # Save image.
            os.makedirs(os.path.join(output_dir, 'results%s' % (r + 1)), exist_ok=True)
            png_filename = os.path.join(os.path.join(output_dir, 'results%s' % (r + 1)), 'example%s.png' % i)
            PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--seed',
        help='random seed',
        default=None,
        type=int
    )
    parser.add_argument(
        '-o',
        '--output_dir',
        help='output directory for results',
        default='resutls',
        type=str
    )
    parser.add_argument(
        '-r',
        '--rounds',
        help='number of iterations to run example',
        default=1,
        type=int
    )
    parser.add_argument(
        '-d',
        '--dataset',
        help='dataset on which model was pre-trained',
        choices=['ffhq', 'celebahq', 'bedrooms', 'cars', 'cats'],
        default='ffhq',
        type=str
    )
    main(**vars(parser.parse_args()))
