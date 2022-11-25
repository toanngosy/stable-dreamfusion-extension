import json
import os

from modules import paths, images, shared

try:
    cmd_dreamfusion_models_path = shared.cmd_opts.dreamfusion_models_path
except:
    cmd_dreamfusion_models_path = None

class DreamfusionConfig(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.name = None
            self.model_name = None
            self.__dict__ = self

        # TODO: what is this?
        def create_new(self):
            pass

        def from_ui(self):
            pass

        def save(self):
            model_path = os.path.dirname(cmd_dreamfusion_models_path) if cmd_dreamfusion_models_path else paths.models_path
            config_file = os.path.join(model_path, "dreamfusion", self.__dict__["model_name"], "df_config.json")
            with open(config_file, "w") as outfile:
                json.dump(self.__dict__, outfile, indent=4)
