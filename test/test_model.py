# %%
import torch
hyperparams = {
    "num_regions": 200,
    "num_res_block": 0,
    "motif_prior": True,
    "num_motif": 637,
    "embed_dim": 768,
    "num_layers": 8,
    "d_model": 768,
    "nhead": 8,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "output_dim": 2,
}

# %%
from timm import create_model
m = create_model('get_finetune_motif', pretrained=False).cuda()

# %%
# random generate one hot encoding of DNA sequence in size (batch_size, num_region, dna_len, 4)
batch_size = 2
num_region = 200
dna_len = 2000
input = torch.randint(0, 2, (batch_size, num_region, dna_len, 4)).cuda().float()
# %%

# %%
for i in range(16):
    output, atac, exp = m(input, torch.zeros((batch_size, num_region)).bool().cuda(), torch.randint(0, 1, (batch_size, num_region, 3)).cuda())
    print(output.shape, atac.shape, exp.shape)
    output.mean().backward()
# %%
