import os  
import sys
sys.path.append(os.getcwd())
import torch   
import lightning.pytorch as pl
import hydra  
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from modelzipper.tutils import *
from utils import *
from captum.attr import IntegratedGradients

def analysis_cov1d_kernel(module):
    weights = module.weight.data.cpu().numpy()
    for i, weight in enumerate(weights):
        plt.plot(weight[0], label=f'Conv Kernel {i}')
    plt.title('Convolution Kernels Weights')
    plt.xlabel('Kernel Size')
    plt.legend()
    plt.show()


class Experiment(pl.LightningModule):
    def __init__(self, model, config, tokenizer=None, state="eval") -> None:
        super(Experiment, self).__init__()
        self.model = model
        self.model.eval()
        self.cfg = config
        self.tokenizer = tokenizer
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        input_ids = batch.pop("input_ids")
        if input_ids.dim() == 3:
            input_ids = input_ids.squeeze(0)
        depth = batch.get('depth').item()
        ctx_length = batch.get('ctx_length').item()
        
        if ctx_length % 1000 != 0:
            pass
        extra_kwargs = {
            "ctx_length": ctx_length,
            "depth": depth
        }
        output = self.model.generate(
            input_ids, min_length=input_ids.size(-1)+10, max_length=input_ids.size(-1)+32, extra_kwargs=extra_kwargs)

        batch['predictions'] = output
        batch['depth'] = depth
        batch['ctx_length'] = ctx_length
        return batch


@hydra.main(config_path='../configs', config_name='analysis_config', version_base='1.1')
def main(config):
    
    print_c(OmegaConf.to_yaml(config), "yellow")
    
    model_root_dir = config.platform.hf_model_path
    data_root_dir = config.platform.dataset_path

    # model_path = os.path.join(model_root_dir, config.model.model_name_or_path)
    # tokenizer_path = os.path.join(model_root_dir, config.model.tokenizer_name_or_path)

    model, tokenizer = get_model_tokenizer(model_root_dir, config.model, analysis=False)
    # load model and tokenizer
    # model = CustomMambaForCausalLM.from_pretrained(
    #     model_path, use_relative_position=False,
    #     dtype=torch.bfloat16, device="cuda", strict=False
    # )
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # load testing data
    data_module = CustomDatamodule(config.task, data_root_dir, tokenizer)
    data_module.setup(stage='predict')

    # load experiment (and model checkpoint)
    experiment = Experiment(model=model, config=config, tokenizer=tokenizer)
    tester = pl.Trainer(devices=config.experiment.device_num, precision="bf16")
    
    #########################
    ## Analysis function Lens
    #########################
    # register hook to get the output of the last layer
    # conv_outputs = []

    # def conv_hook_fn(module, input, output):
    #     conv_outputs.append(output.clone().detach())

    # # register hook to get the output of the last layer
    # hook = model.backbone.layers[-1].mixer.conv1d.register_forward_hook(conv_hook_fn)

    b_t = time.time()
    predictions = tester.predict(
        model=experiment,
        dataloaders=data_module.predict_dataloader(),
        return_predictions=True,
        ckpt_path=config.model.ckpt_path if config.model.load_model_state_dict else None
    )

    print_c(f"======= prediction end, begin to post process and save =======", "magenta")
    
    save_path = os.path.join(config.platform.result_path, f"{config.experiment.results_save_dir}/predictions.pkl")
    auto_save_data(predictions, save_path)
    print_c(f"save predictions to {save_path}, total cost time: {time.time() - b_t}", "magenta")

if __name__ == '__main__':
    main()