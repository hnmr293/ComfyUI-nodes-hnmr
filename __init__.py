from .text import Text
from .number import Integer, Float
from .sampler import SamplerName, SchedulerName
from .cliptextencode2 import CLIPTextEncode2
from .randomlatent import RandomLatentImage
from .vae import VAEDecodeBatched, VAEEncodeBatched
from .sample import KSamplerSetting, KSamplerOverrided, KSamplerXYZ
from .model import StateDictLoader, Dict2Model, StateDictMerger, StateDictMergerBlockWeighted
from .image import GridImage

NODE_CLASS_MAPPINGS = {
    # basic nodes
    
    ## text output
    'Text': Text,
    
    ## integer output
    'Integer': Integer,
    
    ## float output
    'Float': Float,
    
    ## sampler selection
    'SamplerName': SamplerName,
    
    ## scheduler selection
    'SchedulerName': SchedulerName,
    
    # conditioning
    
    ## same as CLIPTextEncode, but the prompt and CLIP are external inputs
    'CLIPTextEncode2': CLIPTextEncode2,
    
    # latent
    
    'RandomLatentImage': RandomLatentImage,
    
    ## pass latents to VAE separately
    'VAEDecodeBatched': VAEDecodeBatched,
    'VAEEncodeBatched': VAEEncodeBatched,
    
    # sampling
    
    ## put parameters for sampler into a dict
    'KSamplerSetting': KSamplerSetting,
    
    ## KSampler with a dict as default setting
    'KSamplerOverrided': KSamplerOverrided,
    
    ## XYZ plotting
    'KSamplerXYZ': KSamplerXYZ,
    
    # loader
    
    ## loads state_dict of the specified checkpoint and returns it
    'StateDictLoader': StateDictLoader,
    
    # model
    
    ## creates model from state_dict loaded by `StateDictLoader`
    'Dict2Model': Dict2Model,
    
    ## merge two (weighted sum) or three (add difference) state_dict
    'StateDictMerger': StateDictMerger,
    
    ## merge block weighted
    ## weights should be specified by Text
    'StateDictMergerBlockWeighted': StateDictMergerBlockWeighted,
    
    # image
    
    ## rearrange images to single image with specified columns and gap
    'GridImage': GridImage,
}
