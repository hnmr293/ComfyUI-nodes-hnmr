import re

re_range = re.compile(r"\s*([+-]?\s*\d+)\s*-\s*([+-]?\s*\d+)(?:\s*\(([+-]\d+)\s*\))?\s*")
re_range_float = re.compile(r"\s*([+-]?\s*\d+(?:.\d*)?)\s*-\s*([+-]?\s*\d+(?:.\d*)?)(?:\s*\(([+-]\d+(?:.\d*)?)\s*\))?\s*")

class Integer:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'value': ('INT', {})
            }
        }
    
    RETURN_TYPES = ('Integer',)
    
    FUNCTION = 'execute'
    
    CATEGORY = 'value'
    
    def execute(self, value: int):
        return (value,)


class Float:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'value': ('FLOAT', {})
            }
        }
    
    RETURN_TYPES = ('Float',)
    
    FUNCTION = 'execute'
    
    CATEGORY = 'value'
    
    def execute(self, value: float):
        return (value,)
