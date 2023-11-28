def predict_expression(rsid, gene, CellMut, center, N, inf_model=inf_model):
    """
    Calculate expression predictions for original and altered cell states based on rsid and gene.

    Args:
    rsid (str): Reference SNP ID.
    gene (str): Gene name.
    CellMut (object): An instance of the CellMut class, containing data and methods for cell mutations.
    center (int): Center position for slicing the data matrix.
    N (int): The size of the slice from the data matrix.

    Returns:
    tuple: A tuple containing expression predictions for the original and altered states.
    """
    # Calculate new motif
    ref = CellMut.mut.df.query('RSID==@rsid').Ref.values[0]
    alt = CellMut.mut.df.query('RSID==@rsid').Alt.values[0]
    new_motif = (CellMut.Alt_input.loc[f'{rsid}_{alt}'].sparse.to_dense().values + 0.01) / \
                (CellMut.Ref_input.loc[f'{rsid}_{ref}'].sparse.to_dense().values + 0.01)

    # Determine start and end indices based on the gene TSS
    gene_tss_info = CellMut.celltype.get_gene_tss(gene)[0]
    start = gene_tss_info.peak_id - center
    end = start + N

    # Get strand information
    strand_idx = gene_tss_info.strand

    # Process original matrix
    original_matrix = CellMut.celltype.input_all[start:end].toarray()
    atac = original_matrix[:, 282].copy()
    original_matrix[:, 282] = 1

    # Process altered matrix
    idx_altered = CellMut.mut.df.query('RSID==@rsid').values[0][0] - start
    print(idx_altered)
    altered_matrix = original_matrix.copy()
    altered_matrix[:, 0:282] = new_motif * altered_matrix[:, 0:282]

    # Create tensors for prediction
    original = torch.Tensor(original_matrix).unsqueeze(0).to(inf_model.device)
    altered = torch.Tensor(altered_matrix).unsqueeze(0).to(inf_model.device)
    seq = torch.randn(1, N, 283, 4).to(inf_model.device)  # Dummy seq data
    tss_mask = torch.ones(1, N).to(inf_model.device)  # Dummy TSS mask
    ctcf_pos = torch.ones(1, N).to(inf_model.device)  # Dummy CTCF positions

    # Predict expression
    _, original_exp = inf_model.predict(original, seq, tss_mask, ctcf_pos)
    _, altered_exp = inf_model.predict(altered, seq, tss_mask, ctcf_pos)

    # Calculate and return the expression predictions
    original_pred = 10 ** (original_exp[0, center, strand_idx].item()) - 1
    altered_pred = 10 ** (altered_exp[0, center, strand_idx].item()) - 1

    return original_pred, altered_pred


def plot_gene_variant_sum_effect(gene, scores):
    # highlight ld variants in red
    df = scores.query('gene==@gene & celltype=="Fetal Astrocyte 1"').groupby('variant').score.sum().sort_values()
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.scatterplot(x=np.arange(len(df)), y=df.values, ax=ax, hue = df.values, palette='RdBu_r', legend=False, hue_norm=(-0.001, 0.001))
    # add y=0 line
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    for i, row in df.reset_index().iterrows():
        # only for top and bottom 5
        if i<5 or i>len(df)-5:
        # if row.variant in ["rs4977756", "rs2383205","rs10811648", "rs10811649",]:
            plt.text(s=row.variant, x=i+0.7, y=row.score, ha='center', va='bottom', alpha=1)

    # remove xticks
    # xlim to 0,20
    # plt.xlim(-1,20)
    plt.xticks([])
    # y label:
    plt.ylabel('Impact score on expression')
    # plt.title('Variants linked to rs4977756', y=1.1)
    plt.tight_layout()
    return fig, ax