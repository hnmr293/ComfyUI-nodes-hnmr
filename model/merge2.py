from typing import Dict, Union, List, Callable, Optional
import torch
import folder_paths
from .loader import Dict2Model
from .merge import block_index, weighted_sum_block, StateDictMergerBlockWeighted
from .iter import iterize_model

from comfy.ldm.models.diffusion.ddpm import LatentDiffusion

class MergedModule(torch.nn.Module):
    
    def __init__(self, name: str, a: torch.nn.Module, b: torch.nn.Module, alpha: Callable[[str],float]):
        super().__init__()
        
        assert hasattr(a, 'weight')
        assert hasattr(b, 'weight')
        
        self._name = name
        self.a = a
        self.b = b
        self.alpha = alpha
        #
        #self.a._apply = self._apply_a
        #self.b._apply = self._apply_b
    
    def forward(self, *args, **kwargs):
        va = self.a(*args, **kwargs)
        vb = self.b(*args, **kwargs)
        a = self.alpha(self._name)
        return (1-a)*va + a*vb
    
    #def _apply_a(self, *args, **kwargs):
    #    torch.nn.Module._apply(self.b, *args, **kwargs)
    #    return torch.nn.Module._apply(self.a, *args, **kwargs)
    #
    #def _apply_b(self, *args, **kwargs):
    #    torch.nn.Module._apply(self.a, *args, **kwargs)
    #    return torch.nn.Module._apply(self.b, *args, **kwargs)
        

ATTR_ALPHAS = 'mbw_alphas'
ATTR_INDEX = 'mbw_index'

def get_current_alpha(model: LatentDiffusion) -> Optional[List[float]]:
    if hasattr(model, ATTR_ALPHAS):
        return getattr(model, ATTR_ALPHAS)[getattr(model, ATTR_INDEX)]
    else:
        return None

def mbw_on_the_fly(
    model_A: LatentDiffusion,
    model_B: LatentDiffusion,
    alphas_list: List[List[float]],
    base_alpha: float,
):
    setattr(model_A, ATTR_ALPHAS, alphas_list)
    setattr(model_A, ATTR_INDEX, 0)
    
    def alpha_fn(name: str):
        block = block_index(name)
        if block is not None and 25 <= block:
            raise ValueError('must not happen')
        
        if block is None:
            return base_alpha
        else:
            index: int = getattr(model_A, ATTR_INDEX)
            return alphas_list[index][block]
    
    def replace(parent_name: str, mod_A: torch.nn.Module, mod_B: torch.nn.Module, alpha: Callable[[str],float]):
        for name, a in list(mod_A.named_children()):
            b = getattr(mod_B, name, None)
            if b is None:
                continue
            
            long_name = f'{parent_name}.{name}' if len(parent_name) != 0 else name
            if not hasattr(a, 'weight') and not hasattr(b, 'weight'):
                replace(long_name, a, b, alpha)
            
            elif hasattr(a, 'weight') and hasattr(b, 'weight'):
                setattr(mod_A, name, MergedModule(long_name, a, b, alpha))
            
            else:
                a_with = 'with' if hasattr(a, 'weight') else 'without'
                b_with = 'with' if hasattr(b, 'weight') else 'without'
                print(f'mismatch: model_A has key {long_name} {a_with} weights, and model_B {b_with} weights.')
    
    replace('', model_A, model_B, alpha_fn)


class StateDictMergerBlockWeightedMulti:
    
    @classmethod
    def INPUT_TYPES(cls):
        d = StateDictMergerBlockWeighted.INPUT_TYPES()
        d['required']['config_name'] = (folder_paths.get_filename_list('configs'), )
        return d

    RETURN_TYPES = ('MODEL','CLIP','VAE')
    
    FUNCTION = 'execute'

    CATEGORY = 'model'

    def execute(
        self,
        model_A: Dict[str,torch.Tensor],
        model_B: Dict[str,torch.Tensor],
        position_ids: str,
        half: str,
        base_alpha: float,
        alphas: str,
        config_name: str,
    ):
        alphas_list = self.get_alphas(alphas)
        
        clip_vae = self.merge_clip_vae(model_A, model_B, base_alpha, position_ids, half)
        
        modelA, clipA, vaeA = self.get_model(model_A, config_name)
        modelB, clipB, vaeB = self.get_model(model_B, config_name)
        
        class WeightLoader(torch.nn.Module):
            pass
        
        w = WeightLoader()
        w.cond_stage_model = clipA.cond_stage_model
        w.first_stage_model = vaeA.first_stage_model
        w.load_state_dict(clip_vae, strict=False)
        
        mbw_on_the_fly(modelA.model, modelB.model, alphas_list, base_alpha)
        
        model_fn = iterize_model(modelA)
        model_fn.clear()
        for index in range(len(alphas_list)):
            def fn(index=index):
                setattr(modelA.model, ATTR_INDEX, index)
                return modelA
            model_fn.append(fn)
        
        return (modelA, clipA, vaeA)
    
    def get_alphas(self, alphas: str):
        alphas_line = [ [ float(x.strip()) for x in line.strip().split(',') if 0 < len(x.strip()) ] for line in alphas.split('\n') ]
        alphas_line = list(filter(lambda vs: len(vs) != 0, alphas_line)) # ignore empty line
        
        for row, line in enumerate(alphas_line, 1):
            if len(line) != 25:
                raise ValueError(f'line {row}: given {len(line)} values, expected 25.')
        
        return alphas_line
    
    def merge_clip_vae(
        self,
        model_A: Dict[str,torch.Tensor],
        model_B: Dict[str,torch.Tensor],
        base_alpha: float,
        position_ids: str,
        half: str
    ):
        def filter_(dic, ss):
            return { k: v for k, v in dic.items() if any(k.startswith(s) for s in ss) }
        
        clip_vae_A = filter_(model_A, ['cond_stage_model', 'first_stage_model'])
        clip_vae_B = filter_(model_B, ['cond_stage_model', 'first_stage_model'])
        
        clip_vae = weighted_sum_block(clip_vae_A, clip_vae_B, base_alpha, [0]*25, position_ids, half)
        return clip_vae
        
    def get_model(self, model: Dict[str,torch.Tensor], config_name: str):
        return Dict2Model().execute(model, config_name)
