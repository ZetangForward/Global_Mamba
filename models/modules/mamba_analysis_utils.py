import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from modelzipper.tutils import *
# input_should_be [x]
def plot1(input_tensor, name=None, type="plt", input_shape="bdl", file_path=None):  
    default_root_path = f"/nvme1/zecheng/modelzipper/projects/state-space-model/analysis/"
    if file_path:
        file_path=file_path
    elif name:
        file_path = os.path.join(default_root_path, f"{name}.png") 
    else:
        file_path = os.path.join(default_root_path, "tmp.png")

    input_tensor = input_tensor.detach().type(torch.float32).cpu()
    # while len(input_tensor.shape)>1: input_tensor=input_tensor.squeeze()
    if input_shape == "bdl":
        input_tensor = input_tensor[0].mean(0).squeeze()
    if input_shape == "bld":
        input_tensor = input_tensor[0].mean(-1).squeeze()

    if type=="sns":
        sns.histplot(input_tensor, kde=True)
    elif type=="plt":
        plt.figure()
        plt.plot(list(range(len(input_tensor))), input_tensor.numpy()) 
    elif type=="bar":
        # if len(input_tensor.shape)>=2: input_tensor=input_tensor.squeeze()
        plt.bar(range(len(input_tensor)), input_tensor)


    plt.title('Fig')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.savefig(file_path)
    log_c(file_path)


if __name__ == "__main__":
    pass
    # plot1(torch.tensor([1,2,3,4]))