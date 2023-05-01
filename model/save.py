import os
import re
from typing import Union
import torch
import folder_paths

class SaveStateDict:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'weights': ('DICT',),
                'filename': ('STRING', { 'multiline': False, 'default': 'merged_model.safetensors' }),
                'overwrite': (['False', 'True'],),
            }
        }
    
    OUTPUT_NODE = True
    
    RETURN_TYPES = ()
    
    FUNCTION = 'execute'
    
    CATEGORY = 'model'
    
    def execute(self, weights: dict, filename: str, overwrite: Union[str,bool]):
        if isinstance(overwrite, str):
            overwrite = overwrite.lower() == 'true'
        
        subdir = os.path.dirname(os.path.normpath(filename))
        basename = os.path.basename(os.path.normpath(filename))
        
        output_dir = folder_paths.get_output_directory()
        full_output_dir = os.path.join(output_dir, subdir)
        
        if os.path.commonpath((output_dir, os.path.realpath(full_output_dir))) != output_dir:
            print('Saving image outside the output folder is not allowed.')
            return {}
        
        full_path = os.path.join(full_output_dir, basename)
        
        if os.path.exists(full_path):
            print(f'{full_path} already exists.')
            if overwrite:
                print(f'overwriting: {full_path}')
            else:
                base, ext = os.path.splitext(basename)
                
                def replace(m: re.Match):
                    x = m.group(0)
                    if len(x) == 0:
                        return '0'
                    else:
                        n = int(x)
                        return str(n + 1)
                
                base_renamed = re.sub(r'\d+$|$', replace, base)
                full_path_renamed = os.path.join(full_output_dir, base_renamed + ext)
                print(f'rename {full_path} -> {full_path_renamed}')
                full_path = full_path_renamed
        
        os.makedirs(full_output_dir, exist_ok=True)
        
        print(f'Saving the state_dict to {full_path}')
        
        ext = os.path.splitext(full_path)[1]
        saver = None
        
        if ext == '.pt' or ext == '.ckpt':
            saver = self.save_torch
        elif ext == '.safetensor' or ext == '.safetensors':
            saver = self.save_safetensors
        else:
            full_path += '.safetensors'
            saver = self.save_safetensors
            
        saver(weights, full_path)
        
        return {}
    
    def save_torch(self, model: dict, path: str):
        torch.save(model, path)
    
    def save_safetensors(self, model: dict, path: str):
        import safetensors.torch
        safetensors.torch.save_file(model, path)
