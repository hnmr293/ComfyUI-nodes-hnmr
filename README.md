# ComfyUI custom nodes

![](./nodes.png)

## Latent nodes

|category|node name|input type|output type|desc.|
| --- | --- | --- | --- | --- |
|latent|RandomLatentImage|`INT`, `INT`, `INT`|`LATENT`|(width, height, batch_size)|
|latent|VAEDecodeBatched|`LATENT`, `VAE`, `INT`|`IMAGE`|VAE decoding with specified batch size|
|latent|VAEEncodeBatched|`IMAGE`, `VAE`, `INT`|`LATENT`|VAE encoding with specified batch size|

## Sampling nodes

|category|node name|input type|output type|desc.|
| --- | --- | --- | --- | --- |
|sampling|KSamplerSetting|`MODEL`, `CONDITIONING`, `CONDITIONING`, `LATENT`|`DICT`|aggregate sampler's setting to single dict|
|sampling|KSamplerOverrided|various|`LATENT`|override sampler's setting defined by `KSamplerSetting`|
|sampling|KSamplerXYZ|various|`LATENT`|generate latents with values|

## Model nodes and Loader nodes

|category|node name|input type|output type|desc.|
| --- | --- | --- | --- | --- |
|loader|StateDictLoader|(model name)|`DICT`|load state_dict|
|model|Dict2Model|`DICT`, (config_file)|`MODEL`|instantiate a model from given state_dict|
|model|StateDictMerger|`DICT`, `DICT`, `FLOAT`|`MODEL`, `CLIP`, `VAE`|merge two or three models|
|model|StateDictMergerBlockWeighted|`DICT`, `DICT`|`DICT`|merge two models with per-block weights|
|model|ModelIter|`MODEL`, `MODEL`|`MODEL`|iterate models|
|model|CLIPlIter|`CLIP`, `CLIP`|`CLIP`|iterate CLIPs|
|model|VAElIter|`VAE`, `VAE`|`VAE`|iterate VAEs|

## Output nodes

|category|node name|input type|output type|desc.|
| --- | --- | --- | --- | --- |
|image|GridImage|||generate single image with specific columns|
