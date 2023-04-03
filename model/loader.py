import folder_paths
from comfy.utils import load_torch_file

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
        
        from comfy import utils, sd
        load_torch_file_org = utils.load_torch_file
        setattr(utils, 'load_torch_file', load_torch_file_hook)
        
        try:
            model, clip, vae = sd.load_checkpoint(config_path, None, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
            assert clip is not None
            assert vae is not None
            return (model, clip, vae)
        finally:
            setattr(sd, 'load_torch_file', load_torch_file_org)

