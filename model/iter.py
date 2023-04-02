from typing import List, Callable
import torch
import tqdm
from comfy.sd import ModelPatcher, CLIP, VAE

class CondForModels(torch.Tensor):
    
    @staticmethod
    def __new__(cls, x, ex, *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs) # type: ignore
    
    def __init__(self, x, ex: List[torch.Tensor], *args, **kwargs):
        super().__init__()
        self.ex = ex

def iterize_model(model: ModelPatcher) -> List[Callable[[],ModelPatcher]]:
    ATTR_NAME = 'iter_fn'
    if not hasattr(model, ATTR_NAME):
        setattr(model, ATTR_NAME, [lambda: model])
    return getattr(model, ATTR_NAME)

def iterize_clip(clip: CLIP) -> List[Callable[[],CLIP]]:
    ATTR_NAME = 'iter_fn'
    if hasattr(clip, ATTR_NAME):
        return getattr(clip, ATTR_NAME)
    
    setattr(clip, ATTR_NAME, [lambda: clip])
    
    old_encode = CLIP.encode
    
    def new_encode(*args, **kwargs):
        xs = []
        clips = getattr(clip, ATTR_NAME)
        for fn in tqdm.tqdm(clips):
            clip_: CLIP = fn()
            if clip_ == clip:
                x = old_encode(clip_, *args, **kwargs)
            else:
                x = clip_.encode(*args, **kwargs)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            xs.append(x)
        return CondForModels(xs[0], xs)
    
    clip.encode = new_encode
    
    return getattr(clip, ATTR_NAME)

def iterize_vae(vae: VAE) -> List[Callable[[],VAE]]:
    ATTR_NAME = 'iter_fn'
    if hasattr(vae, ATTR_NAME):
        return getattr(vae, ATTR_NAME)
    
    setattr(vae, ATTR_NAME, [lambda: vae])
    
    old_decode = VAE.decode
    
    def new_decode(*args, **kwargs):
        xs = []
        vaes = getattr(vae, ATTR_NAME)
        for fn in tqdm.tqdm(vaes):
            vae_: VAE = fn()
            if vae_ == vae:
                x = old_decode(vae_, *args, **kwargs)
            else:
                x = vae_.decode(*args, **kwargs)
            if x.dim() == 3:
                x = x.unsqueeze(0)
            xs.append(x)
        return torch.cat(xs)
    
    vae.decode = new_decode
    
    return getattr(vae, ATTR_NAME)


class ModelIter:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model1': ('MODEL', ),
                'model2': ('MODEL', )
            }
        }
    
    RETURN_TYPES = ('MODEL',)
    
    FUNCTION = 'execute'

    CATEGORY = 'model'

    def execute(self, model1, model2):
        fns = iterize_model(model1)
        fns.append(lambda: model2)
        return (model1,)


class CLIPIter:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'clip1': ('CLIP', ),
                'clip2': ('CLIP', )
            }
        }
    
    RETURN_TYPES = ('CLIP',)
    
    FUNCTION = 'execute'

    CATEGORY = 'model'

    def execute(self, clip1, clip2):
        fns = iterize_clip(clip1)
        fns.append(lambda: clip2)
        return (clip1,)


class VAEIter:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'vae1': ('VAE', ),
                'vae2': ('VAE', )
            }
        }
    
    RETURN_TYPES = ('VAE',)
    
    FUNCTION = 'execute'

    CATEGORY = 'model'

    def execute(self, vae1, vae2):
        fns = iterize_vae(vae1)
        fns.append(lambda: vae2)
        return (vae1,)
