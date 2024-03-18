#%%
import torch
import torch.nn.functional as F
from torch.distributions import Multinomial
#%%
def multinomial_nll(true_counts, logits):
    """
    Compute the multinomial negative log-likelihood in PyTorch
    Args:
      true_counts (Tensor): observed count values
      logits (Tensor): predicted logit values
    """
    true_counts = true_counts.reshape(-1, true_counts.size(-1))
    logits = logits.reshape(-1, logits.size(-1))
    # Creating a Multinomial distribution in PyTorch
    log_prob_sum = 0
    for i in range(true_counts.size(0)):
        dist = Multinomial(total_count=int(true_counts[i].sum()), logits=logits[i])
        # Calculating the log probability and then the negative log-likelihood
        log_prob = dist.log_prob(true_counts[i])
        log_prob_sum += log_prob
    
    nll = -torch.sum(log_prob_sum) / true_counts.size(0)
    return nll

# %%
true_counts = torch.randn((64,1,1000)).abs()
logits = torch.randn(64,1,1000).abs()
# %%
multinomial_nll(true_counts, logits)
# %%
