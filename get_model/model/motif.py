import numpy as np


def parse_meme_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    motifs = []
    current_motif = None
    alphabet = lines[2].split()[1]
    # skip the first 10 lines
    lines = lines[9:]
    for i, line in enumerate(lines):
        if line.startswith("MOTIF"):
            if current_motif is not None:
                motifs.append(current_motif)
            current_motif = {"name": line.split()[1], "letter_prob_matrix": []}
        elif current_motif is not None and line.startswith("letter-probability matrix"):
            letter_prob_matrix = []
            current_motif["width"] = int(line.split()[5])
            for j in range(current_motif["width"]):
                row = list(map(float, lines[i + j + 1].split()))
                letter_prob_matrix.append(row)
            current_motif["letter_prob_matrix"] = np.array(letter_prob_matrix)
            current_motif["width"] = len(letter_prob_matrix)
            if current_motif["width"] < 29:
                current_motif["letter_prob_matrix"] = np.concatenate(
                    (   
                        np.zeros(((29 - current_motif["width"])//2, 4)),
                        current_motif["letter_prob_matrix"],
                        np.zeros((29 - current_motif["width"] - (29 - current_motif["width"])//2, 4)),
                    ),
                    axis=0,
                )
    if current_motif is not None:
        # pad to 29, 4
        motifs.append(current_motif)

    motifs = np.stack([motif["letter_prob_matrix"] for motif in motifs], axis=0)
    return motifs