import re
from itertools import product
from typing import Callable, List, Dict, Any, Union, Tuple, cast
import torch
import model_management # type: ignore
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

def get_noise(seeds: List[int], latent_image: torch.Tensor, disable_noise: bool):
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
            noise_ = torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=torch.manual_seed(s), device="cpu")
            noises.append(noise_)
            latents.append(latent_image)
    
    return torch.cat(noises), torch.cat(latents)

def get_cfg(noises: torch.Tensor, latent_image: torch.Tensor, cfgs: List[float]):
    # batch_size = noises.shape[0] * len(cfgs)
    ns = [noises] * len(cfgs)
    lat = [latent_image] * len(cfgs)
    cf = torch.FloatTensor(cfgs * noises.shape[0])
    return torch.cat(ns), torch.cat(lat), cf[...,None,None,None]

def process_cond(
    conds: List[List[Union[torch.Tensor,dict]]],
    control_nets: Union[list,None],
    noise_shape0: int,
    device
):
    conds_copy = []
    for p in conds:
        t: torch.Tensor = p[0] # type: ignore
        if t.shape[0] < noise_shape0:
            t = torch.cat([t] * noise_shape0)
        t = t.to(device)
        if control_nets is not None and 'control' in p[1]:
            control_nets += [p[1]['control']] # type: ignore
        conds_copy += [[t] + p[1:]]
    return conds_copy

def process_cond_for_models(
    conds: List[List[Union[torch.Tensor,CondForModels,dict]]],
    control_nets: list,
    noise_shape0: int,
    device
):
    assert (
        all(isinstance(p[0], CondForModels) for p in conds) 
        or not any(isinstance(p[0], CondForModels) for p in conds)
    )
    
    if isinstance(conds[0][0], CondForModels):
        sizes = set( len(cast(CondForModels, p[0]).ex) for p in conds )
        assert len(sizes) == 1, f'number of conditions: {sizes}'
        size = sizes.pop()
        
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
        #  + [ [ condA_for_model2, dictA ], [ condB_for_model2, dictB ], ... ]
        #  ...
        # 
        
        result = []
        for model_index in range(size):
            cs = []
            for cond_for_models in conds:
                c: CondForModels = cond_for_models[0] # type: ignore
                rest = cond_for_models[1:]
                cond = c.ex[model_index]
                cs.append([cond, *rest])
            result.append(process_cond(
                cs,
                control_nets if model_index == 0 else None,
                noise_shape0,
                device
            ))
        return result
    
    else:
        return [ process_cond(conds, control_nets, noise_shape0, device) ]
    

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
    latent_image = latent["samples"]
    noise_mask = None
    device = model_management.get_torch_device()

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
    
    noise, latent_image = get_noise(seed, latent_image, disable_noise)
    noise, latent_image, cfg_ = get_cfg(noise, latent_image, cfg)
    
    if "noise_mask" in latent:
        noise_mask = latent['noise_mask']
        noise_mask = torch.nn.functional.interpolate(noise_mask[None,None,], size=(noise.shape[2], noise.shape[3]), mode="bilinear")
        noise_mask = noise_mask.round()
        noise_mask = torch.cat([noise_mask] * noise.shape[1], dim=1)
        noise_mask = torch.cat([noise_mask] * noise.shape[0])
        noise_mask = noise_mask.to(device)

    noise = noise.to(device)
    latent_image = latent_image.to(device)
    cfg_ = cfg_.to(device)

    control_nets = []
    positive_copies = process_cond_for_models(positive, control_nets, noise.shape[0], device)
    negative_copies = process_cond_for_models(negative, control_nets, noise.shape[0], device)

    control_net_models = []
    for x in control_nets:
        control_net_models += x.get_control_models()

    samplers: List[Dict[str,Any]] = []
    for (model_index, model_fn), sampler_name_, scheduler_, steps_ in product(enumerate(iterize_model(model)), sampler_name, scheduler, steps):
        if sampler_name_ not in comfy.samplers.KSampler.SAMPLERS:
            raise ValueError(f'unknown sampler name: {sampler_name_}')
        if scheduler_ not in comfy.samplers.KSampler.SCHEDULERS:
            raise ValueError(f'unknown scheduler name: {scheduler_}')
        samplers.append(dict(
            model_index=model_index,
            model=model_fn,
            steps=steps_,
            device=device,
            sampler=sampler_name_,
            scheduler=scheduler_,
            denoise=denoise,
        ))
    
    all_samples: List[torch.Tensor] = []
    for sampler_args in samplers:
        model_ = sampler_args['model']()
        model_management.load_model_gpu(model_)
        model_management.load_controlnet_gpu(control_net_models)
        sampler_args['model'] = model_.model
        
        model_index = sampler_args.pop('model_index')
        positive_copy = positive_copies[model_index % len(positive_copies)]
        negative_copy = negative_copies[model_index % len(negative_copies)]
        
        sampler = comfy.samplers.KSampler(**sampler_args)
        print(f'XYZ sampler=model@{model_index}/{sampler.sampler}/{sampler.scheduler} {sampler.steps}steps')
        
        alphas = merge2.get_current_alpha(model_.model)
        if alphas is not None:
            print(f'alpha = {alphas}')
        
        samples = sampler.sample(noise, positive_copy, negative_copy, cfg=cfg_, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask)
        samples = samples.cpu()
        all_samples.append(samples)
    for c in control_nets:
        c.cleanup()

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
    