import colorsys
from io import StringIO
import csv
from typing import Optional
import torch
import torchvision.transforms.functional
import PIL.Image
import einops

class LatentToImage:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'samples': ('LATENT',),
                'clamp': ('FLOAT', { 'default': 5.0, 'min': 0.1, 'max': 100.0, 'step': 0.01, }),
            },
            #'optional': {
            #}
        }
    
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'execute'

    CATEGORY = 'latent'
    
    def execute(
        self,
        samples: dict,
        clamp: float,
    ):
        s: torch.Tensor = samples['samples']
        B, C, H, W = s.shape
        assert C == 4
        
        clamp = abs(float(clamp))
        
        s = s.abs().clamp(min=0.0, max=clamp) / clamp
        
        images = []
        for b in range(B):
            for c in range(C):
                t = s[b,c,:,:]
                #image = torchvision.transforms.functional.to_pil_image(t, mode='L')
                #images.append(images)
                rgb = torch.dstack([t,t,t])
                images.append(rgb.unsqueeze_(0))
                # (H,W) -> (B,H,W,C)
        
        return (torch.cat(images),)
        
class LatentToHist:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'samples': ('LATENT',),
                'min_auto': (['Auto', 'Specified'],),
                'min_value': ('FLOAT', { 'default': -5.0, 'min': -100.0, 'max': 0.0, 'step': 0.01, }),
                'max_auto': (['Auto', 'Specified'],),
                'max_value': ('FLOAT', { 'default': 5.0, 'min': 0.0, 'max': 100.0, 'step': 0.01, }),
                'bin_auto': (['Auto', 'Specified'],),
                'bin_count': ('INT', { 'default': 10, 'min': 3, 'max': 1000, 'step': 1, }),
                'ymax_auto': (['Auto', 'Specified'],),
                'ymax': ('FLOAT', { 'default': 1.0, 'min': 0.01, 'max': 1.0, 'step': 0.01, }),
            },
        }
    
    RETURN_TYPES = ('IMAGE', 'STRING')
    FUNCTION = 'execute'

    CATEGORY = 'latent'
    
    def execute(
        self,
        samples: dict,
        min_auto: str,
        min_value: float,
        max_auto: str,
        max_value: float,
        bin_auto: str,
        bin_count: int,
        ymax_auto: str,
        ymax: float,
    ):
        s: torch.Tensor = samples['samples']
        B, C, H, W = s.shape
        assert C == 4
        
        ss = s.view((B,C,-1))
        
        def is_auto(v: str):
            return v.lower() == 'auto'
        
        min_ = torch.min(ss).item() if is_auto(min_auto) else float(min_value)
        max_ = torch.max(ss).item() if is_auto(max_auto) else float(max_value)
        bins = 10 if is_auto(bin_auto) else int(bin_count)
        
        assert min_ < max_
        assert 3 <= bins
        
        hists = [ self.hist(ss[b,:,:], min_, max_, bins) for b in range(B) ]
        
        if is_auto(ymax_auto):
            ymax = max([ torch.max(hist).item() for batch_hists in hists for hist, bin_edges in batch_hists ])
            ymax += 0.01
        else:
            ymax = float(ymax)
        
        sio = StringIO()
        image_tensors = list(self.plot(hists, ymax, sio))
        
        images = torch.cat(image_tensors)
        
        return (images, sio.getvalue())
    
    def hist(self, batch_data: torch.Tensor, min_: float, max_: float, bins: int):
        assert batch_data.dim() == 2
        
        C, N = batch_data.shape
        
        assert C == 4
        
        hists = []
        
        for c in range(C):
            hist, bin_edges = torch.histogram(batch_data[c,:], bins=bins, range=(min_, max_))
            hist.div_(N)
            hists.append((hist, bin_edges))
        
        return hists
    
    def plot(self, hists: list, ymax: float, sio: Optional[StringIO] = None):
        try:
            from matplotlib import pyplot as plt
        except Exception as e:
            raise RuntimeError('LatentToHist requires matplotlib. Please install it.')
        
        if sio is not None:
            writer = csv.writer(sio)
            writer.writerow(['ch', 'value', 'degree'])
        else:
            writer = None
        
        def color(c: int, C: int):
            h = c / C
            return colorsys.hsv_to_rgb(h, 1.0, 1.0)
        
        W = torch.FloatTensor([0.5,0.5]).reshape(1,1,2) # out_ch,in_ch/group,kW
        
        for batch_hists in hists:
            C = len(batch_hists)
            
            fig, ax = plt.subplots(1, 1, figsize=(6,6))
            plots = []
            for c, (hist, bin_edges) in enumerate(batch_hists):
                bin_edges = bin_edges.view((1,1,-1)) # batch,in_ch,iW
                W = W.to(bin_edges.device)
                x = torch.nn.functional.conv1d(bin_edges, W).squeeze()
                
                assert x.shape == hist.shape
                
                plot = ax.plot(
                    x, hist, color=color(c,C),
                    linewidth=2, marker='o', markersize=4,
                )
                
                plots.append(plot[0])
                if writer is not None:
                    writer.writerows([
                        # ch, v, d
                        [c, x[i].item(), hist[i].item()]
                        for i in range(x.size(-1))
                    ])
            
            ax.legend(plots, [f'ch={c}' for c in range(C)], loc=2)
            ax.set_ylim(0, ymax)
            ax.grid(visible=True)
            fig.canvas.draw()

            image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb()) # type: ignore

            plt.close(fig)

            image = image.resize((512, 512))

            image_tensor = torchvision.transforms.functional.to_tensor(image)
            # (C, H, W)

            image_tensor = einops.rearrange(image_tensor, 'c h w -> h w c').unsqueeze(0)
            
            if sio is not None:
                sio.flush()
            
            yield image_tensor
