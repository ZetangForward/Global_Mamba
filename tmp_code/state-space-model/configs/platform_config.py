class PlatformConfig:
    def __init__(self, platform_name='amax_a100') -> None:
        self.platform_name = platform_name
        self.cfg = self.return_config(platform_name)

    def return_config(self, platform_name):
        if "amax" in platform_name.lower():  
            if "a100" in platform_name.lower():
                return self.amax_a100()
            
            elif "3090" in platform_name.lower():
                return self.amax_3090()
            
        elif "langchao" in platform_name.lower():
            return self.langchao()

        elif "hitsz" in platform_name.lower():
            return self.hitsz()

        elif "step" in platform_name.lower():
            return self.step()

        elif "h20" in platform_name.lower():
            return self.h20()

    def hitsz(self):
        return {
            "name":"hitsz",
            "hf_model_path": "/UNICOMFS/hitsz_khchen_4/zecheng/hf_models",
            "dataset_path": "/UNICOMFS/hitsz_khchen_4/zecheng/data",
            "exp_path": "/UNICOMFS/hitsz_khchen_4/zecheng/ckpt",
            "result_path": "/UNICOMFS/hitsz_khchen_4/zecheng/evaluation",
        }

    def step(self):
        return {
            "name":"step.ai",
            "hf_model_path": "/vepfs/wcf/G/zecheng/hf_models",
            "dataset_path": "/vepfs/wcf/G/zecheng/data",
            "exp_path": "/vepfs/wcf/G/zecheng/ckpt",
            "result_path": "/vepfs/wcf/G/zecheng/evaluation",
        }

    def langchao(self):
        return {
            "name":"langchao_suda",
            "hf_model_path": "/public/home/ljt/hf_models",
            "dataset_path": "/public/home/ljt/tzc/data",
            "exp_path": "/public/home/ljt/tzc/ckpt",
            "result_path": "/public/home/ljt/tzc/evaluation",
        }
   
    def amax_a100(self):
        return {
            "name":"amax_a100",
            "hf_model_path": "/nvme/hf_models",
            "dataset_path": "/nvme1/zecheng/data",
            "exp_path": "/nvme1/zecheng/ckpt",
            "result_path": "/nvme1/zecheng/evaluation",
        }
   
    def amax_3090(self):
        return {
            "name":"amax_3090",
            "hf_model_path": "/opt/data/private/hf_models",
            "dataset_path": "/opt/data/private/zecheng/data",
            "exp_path": "/opt/data/private/zecheng/ckpt",
            "result_path": "/opt/data/private/zecheng/evaluation",
        }

    def h20(self):
        return {
            "name": "h20",
            "hf_model_path": "/nvme/hf_models",
            "dataset_path": "/nvme/ywj/data",
            "exp_path": "/nvme/ywj/ckpt",
            "result_path": "/nvme/ywj/evaluation",
        }
