class Text:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'text': ('STRING', {
                    'multiline': True,
                })
            }
        }
    
    RETURN_TYPES = ('TEXT',) # currently no widgets can receive 'STRING' inputs...
    
    FUNCTION = 'execute'
    
    CATEGORY = 'value'
    
    def execute(self, text: str):
        return (text,)
