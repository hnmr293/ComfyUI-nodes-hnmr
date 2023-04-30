import torch
from comfy_extras.nodes_post_processing import Blend

class Blend2(Blend):
    
    @classmethod
    def INPUT_TYPES(cls):
        original = Blend.INPUT_TYPES().copy()
        
        new_blend_modes = [
            'compare_light',
            'compare_dark',
            'compare_color_light',
            'compare_color_dark',
            'abs_diff',
        ]
        
        original['required']['blend_mode'][0].extend(new_blend_modes)
        return original

    def blend_mode(self, img1, img2, mode):
        #B, H, W, C = img1.shape
        
        if mode == 'compare_light':
            return torch.where(img1 < img2, img2, img1)
        elif mode == 'compare_dark':
            return torch.where(img1 < img2, img1, img2)
        elif mode == 'compare_color_light':
            return torch.where(
                torch.mean(img1, dim=-1, keepdim=True) < torch.mean(img2, dim=-1, keepdim=True),
                img2, img1
            )
        elif mode == 'compare_color_dark':
            return torch.where(
                torch.mean(img1, dim=-1, keepdim=True) < torch.mean(img2, dim=-1, keepdim=True),
                img1, img2
            )
        elif mode == 'abs_diff':
            return torch.abs(img1 - img2)
        else:
            return super().blend_mode(img1, img2, mode)
