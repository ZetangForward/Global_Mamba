class PlatformConfig:
    def __init__(self, platform_name='amax_a100') -> None:
        self.platform_name = platform_name
        self.cfg = self.return_config(platform_name)

    def return_config(self, platform_name):
        return {
            "name":"your_platform_name",
            "hf_model_path": "/path/hf_models",
            "dataset_path": "/path/data",
            "exp_path": "/path/ckpt",
            "result_path": "/path/evaluation",
        }


