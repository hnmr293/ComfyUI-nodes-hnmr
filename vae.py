from collections import defaultdict
from typing import Dict, List
import torch
from tqdm import trange
from .model.iter import try_get_iter

class VAEDecodeBatched:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", ),
                "vae": ("VAE", ),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 32,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    CATEGORY = "latent"

    def decode(self, vae, samples, batch_size: int):
        s = samples['samples']
        n = s.shape[0]
        
        iters = try_get_iter(vae)
        if iters is None:
            vae_num = 1
        else:
            vae_num = len(iters)
        
        vae_results: Dict[int,List[torch.Tensor]] = defaultdict(lambda: [])
        
        for i in trange(0, n, batch_size):
            e = min([i+batch_size, n])
            t = s[i:e, ...]
            v = vae.decode(t)
            
            vaes = torch.chunk(v, vae_num)
            
            for vn, vv in enumerate(vaes):
                vae_results[vn].append(vv)
        
        results = []
        for k in sorted(vae_results.keys()):
            v = vae_results[k]
            results.extend(v)
            
        vs = torch.cat(results).contiguous()
        return (vs,)


class VAEEncodeBatched:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixels": ("IMAGE", ),
                "vae": ("VAE", ),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 32,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "latent"

    def encode(self, vae, pixels, batch_size: int):
        n = pixels.shape[0]
        x = (pixels.shape[1] // 64) * 64
        y = (pixels.shape[2] // 64) * 64
        if pixels.shape[1] != x or pixels.shape[2] != y:
            pixels = pixels[:,:x,:y,:]
        
        pixels = pixels[:,:,:,:3]
        
        results = []
        for i in trange(0, n, batch_size):
            e = max([i+batch_size, n])
            t = pixels[i:e, ...]
            v = vae.encode(t)
            results.append(v)
        
        vs = torch.cat(results)
        return ({"samples":vs}, )
