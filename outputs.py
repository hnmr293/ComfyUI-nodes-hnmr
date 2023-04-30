import os
import folder_paths

class SaveText:
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'filename_prefix': ('STRING', { 'default': 'ComfyUI' }),
                'ext': ('STRING', { 'default': 'txt' }),
                'text': ('STRING', { 'multiline': True, 'default': '' }),
            }
        }
    
    OUTPUT_NODE = True
    
    RETURN_TYPES = ()
    
    FUNCTION = 'execute'
    
    CATEGORY = 'utils'
    
    def execute(self, filename_prefix: str, ext: str, text: str):
        def map_filename(filename):
            prefix_len = len(os.path.basename(filename_prefix))
            prefix = filename[:prefix_len + 1]
            try:
                digits = int(filename[prefix_len + 1:].split('_')[0])
            except:
                digits = 0
            return (digits, prefix)
        
        subfolder = os.path.dirname(os.path.normpath(filename_prefix))
        filename = os.path.basename(os.path.normpath(filename_prefix))

        full_output_folder = os.path.join(self.output_dir, subfolder)

        if os.path.commonpath((self.output_dir, os.path.abspath(full_output_folder))) != self.output_dir:
            print("Saving image outside the output folder is not allowed.")
            return {}

        try:
            counter = max(filter(lambda a: a[1][:-1] == filename and a[1][-1] == "_", map(map_filename, os.listdir(full_output_folder))))[0] + 1
        except ValueError:
            counter = 1
        except FileNotFoundError:
            os.makedirs(full_output_folder, exist_ok=True)
            counter = 1

        if ext is None or len(ext) == 0 or ext == '.':
            ext = '.txt'
        if not ext.startswith('.'):
            ext = '.' + ext
        
        file = f"{filename}_{counter:05}_{ext}"
        with open(os.path.join(full_output_folder, file), 'w') as io:
            io.write(text)
        counter += 1
        
        return {}
