class CLIPTextEncode2:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP", ), "text": ("TEXT",)}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    def encode(self, clip, text):
        return ([[clip.encode(text), {}]], )
