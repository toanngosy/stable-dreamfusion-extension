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
        self.revision = None
        self.__dict__ = self

    # TODO: what is this?
    def create_new(self, name, scheduler, src, total_steps):
        #name = images.sanitize_filename_part(name, True)
        self.model_name = name
        self.scheduler = scheduler
        self.src = src
        self.total_steps = total_steps
        self.revision = total_steps
        return self

    def from_ui(self,
                model_dir,
                pretrained_vae_name_or_path,
                half_model,
                text,
                negative,
                o,
                o2,
                test,
                save_mesh,
                eval_interval,
                workspace,
                guidance,
                seed,
                iters,
                lr,
                ckpt,
                cuda_ray,
                max_steps,
                num_steps,
                upsample_steps,
                update_extra_interval,
                max_ray_batch,
                albedo,
                albedo_iters,
                uniform_sphere_rate,
                bg_radius,
                density_thresh,
                fp16,
                backbone,
                sd_version,
                w,
                h,
                jitter_pose,
                bound,
                dt_gamma,
                min_near,
                radius_range,
                fovy_range,
                dir_text,
                suppress_face,
                angle_overhead,
                angle_front,
                lambda_entropy,
                lambda_opacity,
                lambda_orient,
                lambda_smooth,
                gui,
                gui_w,
                gui_h,
                gui_radius,
                gui_fovy,
                gui_light_theta,
                gui_light_phi,
                max_spp
                ):
        base_dict = {}
        try:
            base_dict = self.from_file(model_dir)
        except:
            print(f"Exception loading model from path: {model_dir}")
        if "revision" not in self.__dict__:
            base_dict["revision"] = 0
        #pretrained_vae_name_or_path = images.sanitize_filename_part(pretrained_vae_name_or_path, True)
        models_path = os.path.dirname(cmd_dreamfusion_models_path) if cmd_dreamfusion_models_path else paths.models_path
        model_dir = os.path.join(models_path, "dreamfusion", model_dir)
        working_dir = os.path.join(model_dir, "working")
        base_dict["pretrained_model_name_or_path"] = working_dir
        
        data = {"pretrained_model_name_or_path": working_dir,
                "model_dir": model_dir,
                "pretrained_vae_name_or_path": pretrained_vae_name_or_path,
                "half_model": half_model,
                "text": text,
                "negative": negative,
                "o": o,
                "o2": o2,
                "test": test,
                "save_mesh": save_mesh,
                "eval_interval": eval_interval,
                "workspace": workspace,
                "guidance": guidance,
                "seed": seed,
                "iters": iters,
                "lr": lr,
                "ckpt": ckpt,
                "cuda_ray":cuda_ray,
                "max_steps":max_steps,
                "num_steps":num_steps,
                "upsample_steps": upsample_steps,
                "update_extra_interval": update_extra_interval,
                "max_ray_batch": max_ray_batch,
                "albedo": albedo,
                "albedo_iters": albedo_iters,
                "uniform_sphere_rate": uniform_sphere_rate,
                "bg_radius": bg_radius,
                "density_thresh": density_thresh,
                "fp16": fp16,
                "backbone": backbone,
                "sd_version": sd_version,
                "w": w,
                "h": h,
                "jitter_pose": jitter_pose,
                "bound": bound,
                "dt_gamma": dt_gamma,
                "min_near": min_near,
                "radius_range": radius_range,
                "fovy_range": fovy_range,
                "dir_text": dir_text,
                "suppress_face": suppress_face,
                "angle_overhead": angle_overhead,
                "angle_front": angle_front,
                "lambda_entropy": lambda_entropy,
                "lambda_opacity": lambda_opacity,
                "lambda_orient": lambda_orient,
                "lambda_smooth": lambda_smooth,
                "gui": gui,
                "gui_w": gui_w,
                "gui_h": gui_h,
                "gui_radius": gui_radius,
                "gui_fovy": gui_fovy,
                "gui_light_theta": gui_light_theta,
                "gui_light_phi": gui_light_phi,
                "max_spp": max_spp
                }
        for key in data:
            base_dict[key] = data[key]
        self.__dict__ = base_dict
        return self.__dict__
        
    def from_file(self, model_name):
        """
        Load config data from UI
        Args:
            model_name: The config to load

        Returns: Dict

        """
        #model_name = images.sanitize_filename_part(model_name, True)
        model_path = os.path.dirname(cmd_dreamfusion_models_path) if cmd_dreamfusion_models_path else paths.models_path
        working_dir = os.path.join(model_path, "dreamfusion", model_name, "working")
        config_file = os.path.join(model_path, "dreamfusion", model_name, "df_config.json")
        try:
            with open(config_file, 'r') as openfile:
                config = json.load(openfile)
                for key in config:
                    self.__dict__[key] = config[key]
                if "revision" not in config:
                    if "total_steps" in config:
                        revision = config["total_steps"]
                    else:
                        revision = 0
                    self.__dict__["revision"] = revision
        except Exception as e:
            print(f"Exception loading config: {e}")
            return None
            pass
        return self.__dict__

    def save(self):
        model_path = os.path.dirname(cmd_dreamfusion_models_path) if cmd_dreamfusion_models_path else paths.models_path
        config_file = os.path.join(model_path, "dreamfusion", self.__dict__["model_name"], "df_config.json")
        with open(config_file, "w") as outfile:
            json.dump(self.__dict__, outfile, indent=4)
