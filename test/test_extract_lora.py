# extract lora weights using minLoRA library
#%%
import minlora
import torch
from get_model.model.model_refactored import GETRegionFinetune, GETRegionFinetuneModelConfig
# %%
state_dict = torch.load('../GETRegionFinetune/gjs55hs9/checkpoints/last.ckpt')
# %%
state_dict['state_dict'].keys()
# %%
from minlora import get_lora_state_dict, get_lora_params, name_is_lora
# %%
lora_state_dict = {}
for name in state_dict['state_dict'].keys():
    if name_is_lora(name):
        print(name)
        lora_state_dict[name] = state_dict['state_dict'][name]
# %%
# save lora weights
torch.save(lora_state_dict, 'lora_state_dict.pth')
# %%
# load original weights
old_state_dict = torch.load('/home/xf2217/Projects/get_checkpoints/Astrocytes_natac/checkpoint-best.pth')
# %%
x= old_state_dict['model']['blocks.0.mlp.fc1.weight'].detach().cpu().numpy()
# %%
y=state_dict['state_dict']['model.encoder.blocks.0.mlp.fc1.parametrizations.weight.original'].detach().cpu().numpy()
# %%
import seaborn as sns
sns.scatterplot(x=x.flatten(), y=y.flatten())
# %%
