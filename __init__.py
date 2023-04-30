from .randomlatent import RandomLatentImage
from .vae import VAEDecodeBatched, VAEEncodeBatched
from .sample import KSamplerSetting, KSamplerOverrided, KSamplerXYZ
from .model.loader import StateDictLoader, Dict2Model
from .model.iter import ModelIter, CLIPIter, VAEIter
from .model.merge import StateDictMerger, StateDictMergerBlockWeighted
from .model.merge2 import StateDictMergerBlockWeightedMulti
from .image.image import GridImage
from .image.latenttoimage import LatentToImage, LatentToHist
from .image.blend_extra import Blend2

NODE_CLASS_MAPPINGS = {
    # latent
    
    'RandomLatentImage': RandomLatentImage,
    
    ## pass latents to VAE separately
    'VAEDecodeBatched': VAEDecodeBatched,
    'VAEEncodeBatched': VAEEncodeBatched,
    
    ## convert latent matrix to images
    'LatentToImage': LatentToImage,
    'LatentToHist': LatentToHist,
    
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
    
    ## iterate two models for KSamplerXYZ
    'ModelIter': ModelIter,
    
    ## iterate two CLIPs for KSamplerXYZ
    'CLIPIter': CLIPIter,
    
    ## iterate two VAEs for KSamplerXYZ
    'VAEIter': VAEIter,
    
    ## merge two (weighted sum) or three (add difference) state_dict
    'StateDictMerger': StateDictMerger,
    
    ## merge block weighted
    ## weights should be specified by Text
    'StateDictMergerBlockWeighted': StateDictMergerBlockWeighted,
    
    ## merge block weighted
    ## weights should be specified by Text
    'StateDictMergerBlockWeightedMulti': StateDictMergerBlockWeightedMulti,
    
    # image
    
    ## extra blend mode
    'ImageBlend2': Blend2,
    
    ## rearrange images to single image with specified columns and gap
    'GridImage': GridImage,
}
