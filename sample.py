import re
from itertools import product
from typing import Callable, List, Dict, Any, Union, Tuple, cast
import torch
import comfy.sample
import comfy.model_management
import comfy.samplers
from nodes import common_ksampler
from comfy.sd import ModelPatcher
from .model.iter import iterize_model, CondForModels
from .model import merge2

re_int = re.compile(r"\s*([+-]?\s*\d+)\s*")
re_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*")
re_range = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\(([+-]\d+)\s*\))?\s*")
re_range_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\(([+-]\d+(?:.\d*)?)\s*\))?\s*")

def frange(start, end, step):
    x = float(start)
    end = float(end)
    step = float(step)
    while x < end:
        yield x
        x += step

def get_noise(seeds: List[int], latent_image: torch.Tensor, disable_noise: bool, skip: int):
    noises: List[torch.Tensor] = []
    latents: List[torch.Tensor] = []
    
    if latent_image.dim() == 3:
        latent_image = latent_image.unsqueeze(0) # add batch dim
    
    if disable_noise:
        noise_ = torch.zeros([len(seeds)]+list(latent_image.size())[-3:], dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        noises.append(noise_)
        latents.extend([latent_image] * (len(seeds) // latent_image.shape[0]))
    else:
        for s in seeds:
            noise_ = comfy.sample.prepare_noise(latent_image, s, skip)
            noises.append(noise_)
            latents.append(latent_image)
    
    return torch.cat(noises), torch.cat(latents)

def get_cfg(noises: torch.Tensor, latent_image: torch.Tensor, cfgs: List[float]):
    # batch_size = noises.shape[0] * len(cfgs)
    ns = [noises] * len(cfgs)
    lat = [latent_image] * len(cfgs)
    cf = torch.FloatTensor(cfgs * noises.shape[0])
    return torch.cat(ns), torch.cat(lat), cf[...,None,None,None]

def process_cond_for_models(
    cond: List[List[Union[torch.Tensor,CondForModels,dict]]],
    model_index: int
):
    """
    select conditioning tensor for the current model
    """
    
    assert (
        all(isinstance(p[0], CondForModels) for p in cond) 
        or not any(isinstance(p[0], CondForModels) for p in cond)
    )
    
    if not isinstance(cond[0][0], CondForModels):
        return cond
    
    sizes = set( len(cast(CondForModels, p[0]).ex) for p in cond )
    assert len(sizes) == 1, f'number of conditions: {sizes}'
    
    size = sizes.pop()
    assert model_index < size
    
    #
    # conds
    #  + [ CondForModels, dictA ]
    #  |       .ex + condA for model1
    #  |           + condA for model2
    #  |           ...
    #  |           L condA for model{size}
    #  + [ CondForModels, dictB ]
    #  |       .ex + condB for model1
    #  |           + condB for model2
    #  |           ...
    #  |           L condB for model{size}
    #  ...
    # 
    # vvv
    # 
    # conds
    #  + [ [ condA_for_model1, dictA ], [ condB_for_model1, dictB ], ... ]
    #  + [ [ condA_for_model2, dictA ], [ condB_for_model2, dictB ], ... ]  <- model_index
    #  ...
    # 
    
    result = []
    
    for c, *rest in cond:
        assert isinstance(c, CondForModels)
        actual_cond = c.ex[model_index]
        result.append([actual_cond, *rest])
    
    return result

def xyz_args(
    model: ModelPatcher,
    samplers: List[str],
    schedulers: List[str],
    steps: List[int],
):
    for (model_index, model_fn), sampler, scheduler, step in product(enumerate(iterize_model(model)), samplers, schedulers, steps):
        if sampler not in comfy.samplers.KSampler.SAMPLERS:
            raise ValueError(f'unknown sampler name: {sampler}')
        if scheduler not in comfy.samplers.KSampler.SCHEDULERS:
            raise ValueError(f'unknown scheduler name: {scheduler}')
        
        yield (
            model_index,
            model_fn,
            step,
            sampler,
            scheduler,
        )


def common_ksampler_xyz(
    model: ModelPatcher,
    seed: Union[int,List[int]],
    steps: Union[int,List[int]],
    cfg: Union[float,List[float]],
    sampler_name: Union[str,List[str]],
    scheduler: Union[str,List[str]],
    positive,
    negative,
    latent,
    denoise=1.0,
    disable_noise=False,
    start_step=None,
    last_step=None,
    force_full_denoise=False
):
    if not isinstance(seed, list):
        seed = [seed]
    
    if not isinstance(steps, list):
        steps = [steps]
    
    if not isinstance(cfg, list):
        cfg = [cfg]
    
    if not isinstance(sampler_name, list):
        sampler_name = [sampler_name]
    
    if not isinstance(scheduler, list):
        scheduler = [scheduler]
    
    latent_image = latent["samples"]
    noise_mask = latent.get('noise_mask', None)

    noise, latent_image = get_noise(seed, latent_image, disable_noise, latent.get('batch_index', 0))
    noise, latent_image, cfg_ = get_cfg(noise, latent_image, cfg)
    
    cfg_ = cfg_.to('cuda')
    
    all_samples: List[torch.Tensor] = []
    for (
        model_index, model_fn, step, sampler, scheduler
    ) in xyz_args(model, sampler_name, scheduler, steps):
        
        current_model = model_fn()
        positive_copy = process_cond_for_models(positive, model_index)
        negative_copy = process_cond_for_models(negative, model_index)
        
        print(f'XYZ sampler=model@{model_index}/{sampler}/{scheduler} {step}steps')
        alphas = merge2.get_current_alpha(current_model.model)
        if alphas is not None:
            print(f'alpha = {alphas}')
        
        samples = comfy.sample.sample(
            current_model, noise, step, cfg_, sampler, scheduler,
            positive_copy, negative_copy, latent_image,
            denoise=denoise, disable_noise=disable_noise,
            start_step=start_step, last_step=last_step,
            force_full_denoise=force_full_denoise, noise_mask=noise_mask
        )
        
        samples = samples.cpu()
        all_samples.append(samples)

    out = latent.copy()
    out["samples"] = torch.cat(all_samples)
    return (out, )


class KSamplerSetting:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model': ('MODEL',),
                'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),
                'steps': ('INT', {'default': 20, 'min': 1, 'max': 10000}),
                'cfg': ('FLOAT', {'default': 8.0, 'min': 0.0, 'max': 100.0}),
                'sampler_name': (comfy.samplers.KSampler.SAMPLERS, ),
                'scheduler': (comfy.samplers.KSampler.SCHEDULERS, ),
                'positive': ('CONDITIONING', ),
                'negative': ('CONDITIONING', ),
                'latent_image': ('LATENT', ),
                'denoise': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
            }
        }

    RETURN_TYPES = ('DICT',)
    
    FUNCTION = 'sample'

    CATEGORY = 'sampling'

    def sample(self, **kwargs):
        return kwargs,


class KSamplerOverrided:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'setting': ('DICT',),
            },
            'optional': {
                'model': ('MODEL',),
                'seed': ('Integer', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),
                'steps': ('Integer', {'default': 20, 'min': 1, 'max': 10000}),
                'cfg': ('Float', {'default': 8.0, 'min': 0.0, 'max': 100.0}),
                'sampler_name': ('SamplerName',),
                'scheduler': ('SchedulerName', ),
                'positive': ('CONDITIONING', ),
                'negative': ('CONDITIONING', ),
                'latent_image': ('LATENT', ),
                'denoise': ('Float', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
            }
        }

    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'sample'

    CATEGORY = 'sampling'

    def sample(self, setting: dict, **kwargs):
        if 'latent_image' in setting:
            setting['latent'] = setting['latent_image']
            del setting['latent_image']
        
        setting.update(kwargs)
        
        return common_ksampler(**setting)

class KSamplerXYZ:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'setting': ('DICT',),
            },
            'optional': {
                'model': ('MODEL',),
                'seed': ('STRING', { 'multiline': True, 'default': '' }),
                'steps': ('STRING', { 'multiline': True, 'default': '' }),
                'cfg': ('STRING', { 'multiline': True, 'default': '' }),
                'sampler_name': ('STRING', { 'multiline': True, 'default': '' }),
                'scheduler': ('STRING', { 'multiline': True, 'default': '' }),
            }
        }

    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'sample'

    CATEGORY = 'sampling'

    def sample(self, setting: dict, **kwargs):
        if 'latent_image' in setting:
            setting['latent'] = setting['latent_image']
            del setting['latent_image']
        
        # ignore empty string
        kwargs = { k: v for k, v in kwargs.items() if not isinstance(v, str) or len(v) != 0 }
        
        setting = { **setting, **kwargs }
        
        if isinstance(setting.get('seed', None), str):
            setting['seed'] = self.parse(setting['seed'], self.parse_int)
            
        if isinstance(setting.get('steps', None), str):
            setting['steps'] = self.parse(setting['steps'], self.parse_int)
        
        if isinstance(setting.get('cfg', None), str):
            setting['cfg'] = self.parse(setting['cfg'], self.parse_float)
        
        if isinstance(setting.get('sampler_name', None), str):
            setting['sampler_name'] = self.parse(setting['sampler_name'], None)
            if len(setting['sampler_name']) == 1:
                setting['sampler_name'] = setting['sampler_name'][0]
        
        if isinstance(setting.get('scheduler', None), str):
            setting['scheduler'] = self.parse(setting['scheduler'], None)
            if len(setting['scheduler']) == 1:
                setting['scheduler'] = setting['scheduler'][0]
        
        for k, v in setting.items():
            if k in kwargs and isinstance(v, (list, tuple)):
                print(f'XYZ {k}: {v}')
        
        return common_ksampler_xyz(**setting) # type: ignore
    
    def parse(self, input: str, cont: Union[Callable[[str],Any],None]):
        vs = [ x.strip() for x in input.split(',') ]
        if cont is not None:
            new_vs = []
            for v in vs:
                new_v = cont(v)
                if isinstance(new_v, list):
                    new_vs += new_v
                else:
                    new_vs.append(new_v)
            vs = new_vs
        return vs
    
    def parse_int(self, input: str):
        m = re_int.fullmatch(input)
        if m is not None:
            return int(m.group(1))
        
        m = re_range.fullmatch(input)
        if m is None:
            raise ValueError(f'failed to process: {input}')
        
        start, end, step = m.group(1), m.group(2), m.group(3)
        if step is None:
            step = 1

        return list(range(int(start), int(end) + 1, int(step)))
    
    def parse_float(self, input: str):
        m = re_float.fullmatch(input)
        if m is not None:
            return float(m.group(1))
        
        m = re_range_float.fullmatch(input)
        if m is None:
            raise ValueError(f'failed to process: {input}')
        
        start, end, step = m.group(1), m.group(2), m.group(3)
        if step is None:
            step = 1.0
        
        return list(frange(float(start), float(end), float(step)))
    