import comfy.samplers

class SamplerName:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'value': (comfy.samplers.KSampler.SAMPLERS,)
            }
        }
    
    RETURN_TYPES = ('SamplerName',)
    
    FUNCTION = 'execute'
    
    CATEGORY = 'value'
    
    def execute(self, value: str):
        return (value,)

class SchedulerName:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'value': (comfy.samplers.KSampler.SCHEDULERS,)
            }
        }
    
    RETURN_TYPES = ('SchedulerName',)
    
    FUNCTION = 'execute'
    
    CATEGORY = 'value'
    
    def execute(self, value: str):
        return (value,)
