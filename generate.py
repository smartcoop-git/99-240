# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from torchvision.utils import save_image
import torchvision.transforms as T
from PIL import Image

import legacy

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--text', help='Text input', type=str, required=True)

def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    text: str,
    class_idx: Optional[int],
    projected_w: Optional[str],
):

    def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    
    textname = text.replace(' ','_')
    os.makedirs('outputs', exist_ok=True)
    createFolder('outputs/' + textname)
    createFolder('outputs/' + textname + '/test_img')
    createFolder('outputs/' + textname + '/gen_img/')
    
    import shutil
    import glob

    '''
    textlist = text.split(',')
    
    
    if len(textlist) > 2:
        
        if len(design) > 1:
            for i in range(len(design)):
                globals()['design{}'.format(i)] = design[i].strip()
        
        trademark = textlist[0].strip()
        industry = textlist[1].strip()
        design = textlist[2].strip()
        
        filepath = glob.glob('./s3_datasets/' + trademark + '/' + industry + '/' + design + '/*.png')
        
        if len(filepath) > 1:
            for i in range(len(filepath)):
                filename = filepath[i].split('/')[-1]
                shutil.copyfile(filepath[i], 'outputs/' + textname + '/test_img/' + filename)
        elif len(filepath) == 1:
            filename = filepath[0].split('/')[-1]
            shutil.copyfile(filepath[0], 'outputs/' + textname + '/test_img/' + filename)
        else:
            print('The design keyword image does not exist.')
        
    elif len(textlist) > 1:
        trademark = textlist[0].strip()
        industry = textlist[1].strip()
        
        filepath = glob.glob('./s3_datasets/' + trademark + '/' + industry + '/**/*.png')

        for i in range(len(filepath)):
            filename = filepath[i].split('/')[-1]
            shutil.copyfile(filepath[i], 'outputs/' + textname + '/test_img/' + filename)
            
    else:
        trademark = textlist[0].strip()
        
        filepath = glob.glob('./s3_datasets/' + trademark + '/**/*.png')

        for i in range(len(filepath)):
            filename = filepath[i].split('/')[-1]
            shutil.copyfile(filepath[i], 'outputs/' + textname + '/test_img/' + filename)
            '''
            
    trademark = text.strip()
        
    filepath = glob.glob('./s3_datasets/' + trademark + '/**/*.png')

    for i in range(len(filepath)):
        filename = filepath[i].split('/')[-1]
        shutil.copyfile(filepath[i], 'outputs/' + textname + '/test_img/' + filename)
    
    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print ('warn: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{text}/proj{idx:02d}.png')
        return

    if seeds is None:
        ctx.fail('--seeds option is required when not using --projected-w')

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'outputs/{textname}/gen_img/{seed}_{textname}.png')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
