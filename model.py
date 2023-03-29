import re
from typing import Dict, Union, List, Callable
import torch
import tqdm
import folder_paths
from comfy.sd import load_torch_file, load_checkpoint

#class ModelName:
#    
#    @classmethod
#    def INPUT_TYPES(cls):
#        return {
#            'required': {
#                'value': (folder_paths.get_filename_list("checkpoints"),)
#            }
#        }
#    
#    RETURN_TYPES = ('STRING',)
#    
#    FUNCTION = 'execute'
#    
#    CATEGORY = 'value'
#    
#    def execute(self, value: str):
#        return (value,)


class StateDictLoader:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'ckpt_name': (folder_paths.get_filename_list("checkpoints"), )
            }
        }
    
    RETURN_TYPES = ('DICT',)
    
    FUNCTION = 'execute'

    CATEGORY = 'loaders'

    def execute(self, ckpt_name: str):
        ckpt_path = folder_paths.get_full_path('checkpoints', ckpt_name)
        sd = load_torch_file(ckpt_path)
        return (sd,)
    

class Dict2Model:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'weights': ('DICT', ),
                'config_name': (folder_paths.get_filename_list('configs'), ),
            }
        }
    
    RETURN_TYPES = ('MODEL', 'CLIP', 'VAE')
    
    FUNCTION = 'execute'

    CATEGORY = 'model'

    def execute(self, weights: dict, config_name: str):
        config_path = folder_paths.get_full_path("configs", config_name)
        
        def load_torch_file_hook(*args, **kwargs):
            return weights
        
        import comfy.sd as sd
        load_torch_file_org = sd.load_torch_file
        setattr(sd, 'load_torch_file', load_torch_file_hook)
        
        try:
            return sd.load_checkpoint(config_path, None, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        finally:
            setattr(sd, 'load_torch_file', load_torch_file_org)


def merge(
    model_A: Dict[str,torch.Tensor],
    model_B: Dict[str,torch.Tensor],
    merge_fn: Callable[[str, torch.Tensor, torch.Tensor], torch.Tensor],
    position_ids: str,
    half: str,
    ignore_keys_only_in_B: bool = False,
):
    result = dict()
    for key in tqdm.tqdm(model_A.keys()):
        if key not in model_B:
            print(f'  key {key} is found in model_A but not model_B')
            result[key] = model_A[key]
            continue
        
        if key.endswith('.position_ids'):
            continue
        
        #print(f'  key : {key}')
        
        val = merge_fn(key, model_A[key], model_B[key])
        
        if half == 'True':
            val = val.half()
        else:
            val = val.float()
        
        result[key] = val
    
    for key in model_B.keys():
        if not ignore_keys_only_in_B and key not in model_A:
            if key.endswith('.position_ids'):
                continue
            
            print(f'  key {key} is not found in model_A but model_B')
            
            val = model_B[key]
            if half == 'True':
                val = val.half()
            else:
                val = val.float()
            
            result[key] = val
    
    print('position_ids')
    if position_ids == 'A':
        position_ids_key = next(x for x in model_A.keys() if '.position_ids' in x)
        position_ids_val = model_A[position_ids_key]
        print(f"  using model_A's one ({position_ids_val.dtype}: {position_ids_val.shape})")
    elif position_ids == 'B':
        position_ids_key = next(x for x in model_B.keys() if '.position_ids' in x)
        position_ids_val = model_B[position_ids_key]
        print(f"  using model_B's one ({position_ids_val.dtype}: {position_ids_val.shape})")
    elif position_ids == 'Reset':
        position_ids_key = next(x for x in model_A.keys() if '.position_ids' in x)
        position_ids_val = torch.LongTensor(list(range(77))).reshape(model_A[position_ids_key].shape)
        print(f"  reset ({position_ids_val.dtype}: {position_ids_val.shape})")
    else:
        raise ValueError('must not happen')
    result[position_ids_key] = position_ids_val
    
    return result


def weighted_sum(
    model_A: Dict[str,torch.Tensor],
    model_B: Dict[str,torch.Tensor],
    alpha1: float,
    alpha2: float,
    position_ids: str,
    half: str,
):
    print('merging ...')
    print('mode: Weighted Sum')
    
    def merge_fn(key, t1, t2):
        return alpha1 * t1 + alpha2 * t2
    
    return merge(model_A, model_B, merge_fn, position_ids, half)

def add_diff(
    model_A: Dict[str,torch.Tensor],
    model_B: Dict[str,torch.Tensor],
    model_C: Dict[str,torch.Tensor],
    alpha: float,
    position_ids: str,
    half: str,
):
    print('merging ...')
    print('mode: Add Difference')
    
    print('X = B - C')
    def merge_fn1(key, t1, t2):
        return t1 - t2
    model_X = merge(model_B, model_C, merge_fn1, 'A', half, ignore_keys_only_in_B=True)
    
    print('A + alpha*X')
    def merge_fn2(key, t1, t2):
        return t1 + t2*alpha
    result = merge(model_A, model_X, merge_fn2, position_ids, half)
    
    return result

re_inp = re.compile(r'\.input_blocks\.(\d+)\.')
re_mid = re.compile(r'\.middle_block\.(\d+)\.')
re_out = re.compile(r'\.output_blocks\.(\d+)\.')

def weighted_sum_block(
    model_A: Dict[str,torch.Tensor],
    model_B: Dict[str,torch.Tensor],
    base_alpha: float,
    alphas: List[float],
    position_ids: str,
    half: str,
):
    print('merging ...')
    print('mode: Block Weighted')
    
    def index(key: str):
        if not key.startswith('model.diffusion_model.'):
            return None
        if 'time_embed' in key:
            return 0
        if '.out.' in key:
            return 24
        m = re_inp.search(key)
        if m: return int(m.group(1))
        m = re_mid.search(key)
        if m: return 12 + int(m.group(1))
        m = re_out.search(key)
        if m: return 13 + int(m.group(1))
        return None
    
    def merge_fn(key, t1, t2):
        weight_index = index(key)
        if weight_index is None:
            alpha = base_alpha
        elif 25 <= weight_index:
            raise ValueError('must not happen')
        else:
            alpha = alphas[weight_index]
        #print(key, alpha)
        return (1.0 - alpha) * t1 + alpha * t2
    
    return merge(model_A, model_B, merge_fn, position_ids, half)


class StateDictMerger:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'model_A': ('DICT',),
                'model_B': ('DICT',),
                'alpha': ('FLOAT', {
                    'default': 0,
                    'min': -1,
                    'max': 2,
                    'step': 0.001
                }),
                'position_ids': (['A', 'B', 'Reset'], ),
                'half': (['True', 'False'], ),
            },
            'optional': {
                'model_C': ('DICT',),
            },
        }
    
    RETURN_TYPES = ('DICT',)
    
    FUNCTION = 'execute'

    CATEGORY = 'model'

    def execute(
        self,
        model_A: Dict[str,torch.Tensor],
        model_B: Dict[str,torch.Tensor],
        alpha: float,
        position_ids: str,
        half: str,
        model_C: Union[Dict[str,torch.Tensor],None] = None,
    ):
        if model_C is None:
            result = weighted_sum(model_A, model_B, 1.0 - alpha, alpha, position_ids, half)
        else:
            result = add_diff(model_A, model_B, model_C, alpha, position_ids, half)
        return (result,)
    

class StateDictMergerBlockWeighted(StateDictMerger):
    
    @classmethod
    def INPUT_TYPES(cls):
        d = StateDictMerger.INPUT_TYPES()
        d['required']['base_alpha'] = d['required']['alpha']
        del d['required']['alpha']
        del d['optional']['model_C']
        d['required']['alphas'] = ('TEXT',)
        return d

    def execute(
        self,
        model_A: Dict[str,torch.Tensor],
        model_B: Dict[str,torch.Tensor],
        position_ids: str,
        half: str,
        base_alpha: float,
        alphas: str,
    ):
        alphas_ = [float(x.strip()) for x in alphas.split(',')]
        if len(alphas_) != 25:
            raise ValueError(f'given {len(alphas_)} values, expected 25.')
        result = weighted_sum_block(model_A, model_B, base_alpha, alphas_, position_ids, half)
        
        return (result,)
