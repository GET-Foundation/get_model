# mixup strategy for genomics data
# basic assumptions:
# 1. the data has two component, peak (batch, num_region_per_sample, motif) and sequence (batch, num_region_per_sample, seq_len, 4)
# 2. the pretrain data has two label, atac