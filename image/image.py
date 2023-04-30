import os
import math
import json
from typing import List
import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from nodes import SaveImage

class GridImage(SaveImage):
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'images': ('IMAGE',),
                'filename_prefix': ('STRING', {'default': 'ComfyUI-Grid'}),
                'x': ('INT', {
                    'default': 1,
                    'min': 1,
                    'max': 64,
                    'step': 1
                }),
                'gap': ('INT', {
                    'default': 0,
                    'min': 0,
                    'max': 32,
                    'step': 1
                }),
            },
            'hidden': {
                'prompt': 'PROMPT',
                'extra_pnginfo': 'EXTRA_PNGINFO'
            },
        }
    
    OUTPUT_NODE = True
    
    RETURN_TYPES = ()
    
    FUNCTION = 'execute'
    
    CATEGORY = 'image'
    
    def execute(self, images: List[torch.Tensor], filename_prefix: str = 'ComfyUI-Grid', x: int = 1, gap: int = 0, prompt=None, extra_pnginfo=None):
        y = max([math.ceil(len(images) / x), 1])
        
        def map_filename(filename):
            prefix_len = len(os.path.basename(filename_prefix))
            prefix = filename[:prefix_len + 1]
            try:
                digits = int(filename[prefix_len + 1:].split('_')[0])
            except:
                digits = 0
            return (digits, prefix)

        subfolder = os.path.dirname(os.path.normpath(filename_prefix))
        filename = os.path.basename(os.path.normpath(filename_prefix))

        full_output_folder = os.path.join(self.output_dir, subfolder)

        if os.path.commonpath((self.output_dir, os.path.realpath(full_output_folder))) != self.output_dir:
            print("Saving image outside the output folder is not allowed.")
            return {}

        try:
            counter = max(filter(lambda a: a[1][:-1] == filename and a[1][-1] == "_", map(map_filename, os.listdir(full_output_folder))))[0] + 1
        except ValueError:
            counter = 1
        except FileNotFoundError:
            os.makedirs(full_output_folder, exist_ok=True)
            counter = 1

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        results = list()
        
        canvas = self.grid_image(images, x, y, gap)
        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        file = f"{filename}_{counter:05}_.png"
        canvas.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
        results.append({
            "filename": file,
            "subfolder": subfolder,
            "type": 'output'
        })
        counter += 1

        return { "ui": { "images": results } }

    def grid_image(self, images: List[torch.Tensor], x: int, y: int, gap: int):
        width, height, _ = images[0].shape
        canvas = Image.new('RGB', (x*(width+gap)-gap, y*(height+gap)-gap), color='black')
        
        for Y in range(y):
            for X in range(x):
                idx = Y * x + X
                if len(images) <= idx:
                    return canvas
                
                image = images[idx]
                
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                
                canvas.paste(img, (X*(width+gap), Y*(height+gap)))
        
        return canvas
