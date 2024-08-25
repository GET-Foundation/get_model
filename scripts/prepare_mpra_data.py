#!/usr/bin/env python3
"""
Prepare MPRA data by inserting elements into a template, scanning motifs,
and generating normalized chunks for further analysis.

This script performs the following steps:
1. Insert elements into a template sequence
2. Scan motifs and generate a motif vector for each element
3. Normalize the motif scan results
4. Split the results into chunks for further processing

Usage:
    python prepare_mpra_data.py config.yaml
"""

import os
import sys
import yaml
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import gc
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import re
from atac_rna_data_processing.io.sequence import DNASequenceCollection
from atac_rna_data_processing.io.nr_motif_v1 import NrMotifV1


def load_config(config_file):
    """Load configuration from YAML file."""
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def insert_elements_to_template(config):
    """Insert elements into the template sequence."""
    template = str(SeqIO.read(config["template_file"], "fasta").seq)
    elements = SeqIO.parse(config["elements_file"], "fasta")

    new_elements = []
    for element in elements:
        new_seq = re.sub("__Enhancer__", str(element.seq), template).upper()
        new_seq = Seq(new_seq)
        new_element = SeqRecord(seq=new_seq, id=element.id.strip(":"), description="")
        new_elements.append(new_element)

    output_file = config["output_with_template"]
    SeqIO.write(new_elements, output_file, "fasta")
    return output_file


def scan_motifs_chunk(chunk, motif_database_path):
    """Scan motifs for a chunk of elements."""
    motifs = NrMotifV1(motif_database_path)
    chunk_collection = DNASequenceCollection(chunk)
    result = chunk_collection.scan_motif(motifs)
    del motifs, chunk_collection
    gc.collect()
    return result

def scan_motifs(config, input_file):
    """Scan motifs for each element and generate a motif vector using ProcessPoolExecutor."""
    elements_with_template = DNASequenceCollection.from_fasta(filename=input_file)
    chunk_size = config.get("chunk_size", 1000)
    chunks = [elements_with_template.sequences[i:i+chunk_size] 
              for i in range(0, len(elements_with_template.sequences), chunk_size)]

    results = []
    with ProcessPoolExecutor(max_workers=config["n_cores"]) as executor:
        futures = [executor.submit(scan_motifs_chunk, chunk, config["motif_database_path"]) 
                   for chunk in chunks]
        for future in futures:
            results.append(future.result())
            gc.collect()

    return pd.concat(results)


def normalize_results(results):
    """Normalize the motif scan results."""
    results = results.sparse.to_dense()
    results["Accessibility"] = 1
    results = results / results.max(0)
    return results.reset_index()


def save_results(results, config):
    """Save the normalized results and split into chunks."""
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Save full results
    full_output = os.path.join(output_dir, "motif_scan_normalized.feather")
    results.to_feather(full_output)

    # Split into chunks and save
    chunk_size = config["chunk_size"]
    results_chunks = [
        results.iloc[i : i + chunk_size] for i in range(0, len(results), chunk_size)
    ]
    for i, chunk in enumerate(results_chunks):
        chunk_output = os.path.join(
            output_dir, f"motif_scan_normalized_chunk{i}.feather"
        )
        chunk.reset_index(drop=True).to_feather(chunk_output)


def main(config_file):
    """Main function to run the MPRA data preparation pipeline."""
    config = load_config(config_file)

    print("Inserting elements into template...")
    elements_with_template = insert_elements_to_template(config)

    print("Scanning motifs...")
    motif_scan_results = scan_motifs(config, elements_with_template)

    print("Normalizing results...")
    normalized_results = normalize_results(motif_scan_results)

    print("Saving results and splitting into chunks...")
    save_results(normalized_results, config)

    print("MPRA data preparation complete!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prepare_mpra_data.py config.yaml")
        sys.exit(1)

    main(sys.argv[1])