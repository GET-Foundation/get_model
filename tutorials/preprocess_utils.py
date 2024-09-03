# This file contains some utility functions for preprocessing the data for the region-level GET model.
# You can use this instead of modules/atac_rna_data_processing to produce the zarr-based data

import os
import subprocess

import numpy as np
import pandas as pd
import zarr
from pyranges import PyRanges as pr
from pyranges import read_gtf


class Gencode(object):
    """Read gencode gene annotation given genome assembly and version,
    returns a pandas dataframe"""

    def __init__(
        self, assembly="hg38", version=40, gtf_dir="../data", exclude_chrs=["chrM", "chrY"]
    ):
        super(Gencode, self).__init__()

        self.assembly = assembly
        self.gtf_dir = gtf_dir  # New parameter for specifying GTF file location

        if self.assembly == "hg38":
            self.url = "http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{version}/gencode.v{version}.basic.annotation.gtf.gz".format(
                version=str(version)
            )
            self.gtf = os.path.join(
                self.gtf_dir,
                "gencode.v{version}.basic.annotation.gtf.gz".format(
                    version=str(version)
                ),
            )
        elif self.assembly == "mm10":
            self.url = "http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_{version}/gencode.v{version}.basic.annotation.gtf.gz".format(
                version=str(version)
            )
            self.gtf = os.path.join(
                self.gtf_dir,
                "gencode.v{version}.basic.annotation.gtf.gz".format(
                    version=str(version)
                ),
            )
        elif self.assembly == "hg19":
            self.url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{version}/GRCh37_mapping/gencode.v{version}lift37.basic.annotation.gtf.gz".format(
                version=str(version)
            )
            self.gtf = os.path.join(
                self.gtf_dir,
                "gencode.v{version}lift37.basic.annotation.gtf.gz".format(
                    version=str(version)
                ),
            )

        if os.path.exists(
            os.path.join(
                self.gtf_dir,
                "gencode.v{version}.{assembly}.feather".format(
                    version=str(version), assembly=self.assembly
                ),
            )
        ):
            self.gtf = pd.read_feather(
                os.path.join(
                    self.gtf_dir,
                    "gencode.v{version}.{assembly}.feather".format(
                        version=str(version), assembly=self.assembly
                    ),
                )
            )
            self.feather_file = os.path.join(
                self.gtf_dir,
                "gencode.v{version}.{assembly}.feather".format(
                    version=str(version), assembly=self.assembly
                ),
            )
        else:
            if os.path.exists(self.gtf):
                self.gtf = read_gtf(self.gtf).as_df()
            else:
                # download gtf to the specified directory
                os.system("wget -P {dir} {url}".format(dir=self.gtf_dir, url=self.url))
                self.gtf = read_gtf(self.gtf).as_df()

            positive = self.gtf[
                (self.gtf.Feature == "transcript") & (self.gtf.Strand == "+")
            ][
                [
                    "Chromosome",
                    "Start",
                    "Start",
                    "Strand",
                    "gene_name",
                    "gene_id",
                    "gene_type",
                ]
            ]
            negative = self.gtf[
                (self.gtf.Feature == "transcript") & (self.gtf.Strand == "-")
            ][
                [
                    "Chromosome",
                    "End",
                    "End",
                    "Strand",
                    "gene_name",
                    "gene_id",
                    "gene_type",
                ]
            ]

            positive.columns = [
                "Chromosome",
                "Start",
                "End",
                "Strand",
                "gene_name",
                "gene_id",
                "gene_type",
            ]
            negative.columns = [
                "Chromosome",
                "Start",
                "End",
                "Strand",
                "gene_name",
                "gene_id",
                "gene_type",
            ]

            self.gtf = (
                pd.concat([positive, negative], axis=0).drop_duplicates().reset_index()
            )
            self.gtf["gene_id"] = self.gtf.gene_id.str.split(".").str[0]
            self.gtf.to_feather(
                os.path.join(
                    self.gtf_dir,
                    "gencode.v{version}.{assembly}.feather".format(
                        version=str(version), assembly=self.assembly
                    ),
                )
            )
            self.feather_file = os.path.join(
                self.gtf_dir,
                "gencode.v{version}.{assembly}.feather".format(
                    version=str(version), assembly=self.assembly
                ),
            )
        self.gtf = self.gtf[~self.gtf.Chromosome.isin(exclude_chrs)]
        # count number of chromosomes per gene and remove the genes with multiple chromosomes
        self.gtf["chrom_count"] = self.gtf.groupby("gene_name")["Chromosome"].transform(
            "nunique"
        )
        self.gtf = self.gtf[self.gtf.chrom_count == 1]
        return


def download_motif(motif_url, index_url):
    """
    Download motif data from a URL and its index.

    This function uses the `wget` command-line tool to download the motif file and its index.

    Args:
        motif_url (str): URL to the motif file.
        index_url (str): URL to the index file.
    """
    subprocess.run(
        ["wget", "-O", "hg38.archetype_motifs.v1.0.bed.gz", motif_url], check=True
    )
    subprocess.run(
        ["wget", "-O", "hg38.archetype_motifs.v1.0.bed.gz.tbi", index_url], check=True
    )


def join_peaks(peak_bed, reference_peaks=None):
    """
    Join peaks from a peak file and a reference peak file.

    This function uses the `bedtools` command-line tool to perform the following operations:
    1. Intersect the peak file with the reference peak file.
    2. Save the final result to a bed file.

    Args:
        peak_bed (str): Path to the peak file.
        reference_peaks (str): Path to the reference peak file.

    Returns:
        str: Path to the output bed file containing the joined peaks.
    """
    if reference_peaks:
        subprocess.run(
            [
                "bedtools",
                "intersect",
                "-a",
                peak_bed,
                "-b",
                reference_peaks,
                "-wa",
                "-u",
                "-o",
                "joint_peaks.bed",
            ],
            check=True,
        )

    else:
        if os.path.exists("join_peaks.bed"):
            os.remove("join_peaks.bed")
        os.symlink(peak_bed, "join_peaks.bed")
    return "join_peaks.bed"


def query_motif(peak_bed, motif_bed):
    """
    Query motif data from a peak file and a motif file.

    This function uses the `tabix` command-line tool to perform the following operations:
    1. Intersect the peak file with the motif file.
    2. Save the final result to a bed file.

    Args:
        peak_bed (str): Path to the peak file.
        motif_bed (str): Path to the motif file.

    Returns:
        str: Path to the output bed file containing the peak motif data.
    """
    subprocess.run(
        ["tabix", motif_bed, "-R", peak_bed],
        stdout=open("query_motif.bed", "w"),
        check=True,
    )
    return "query_motif.bed"


def get_motif(peak_file, motif_file):
    """
    Get motif data from a peak file and a motif file.

    This function uses the `bedtools` command-line tool to perform the following operations:
    1. Intersect the peak file with the motif file.
    2. Group the intersected data by peak and motif.
    3. Sum the scores for overlapping peaks and motifs.
    4. Sort the resulting data by chromosome, start, end, and motif.
    5. Save the final result to a bed file.

    Args:
        peak_file (str): Path to the peak file.
        motif_file (str): Path to the motif file.

    Returns:
        str: Path to the output bed file containing the peak motif data.
    """

    cmd = f"""
    ASSEMBLY="hg38"
    MEM="12G"
    CPU="2"
    ATAC_PEAK_FILE="{peak_file}"
    ATAC_MOTIF_FILE="{motif_file}"
    OUTPUT_PATH="get_motif.bed"

    OUTPUT_BASE=$(basename "$OUTPUT_PATH" .bed)

    awk -v OUTPUT_BASE="$OUTPUT_BASE" '{{OFS="\\t"; print $1,$2,$3 > OUTPUT_BASE"."$1".peak.bed"}}' "$ATAC_PEAK_FILE"
    awk -v OUTPUT_BASE="$OUTPUT_BASE" '{{print > OUTPUT_BASE"."$1".motif.bed"}}' "$ATAC_MOTIF_FILE"

    for CHR in $(ls "${{OUTPUT_BASE}}."*".motif.bed" | grep -v random | grep -v chrY | cut -d'.' -f2); do
        bedtools intersect -a "${{OUTPUT_BASE}}.${{CHR}}.peak.bed" -b "${{OUTPUT_BASE}}.${{CHR}}.motif.bed" -wa -wb | cut -f1,2,3,7,8,10 > "${{OUTPUT_BASE}}.${{CHR}}.peak_motif.bed"
        sort -k1,1 -k2,2n -k3,3n -k4,4 "${{OUTPUT_BASE}}.${{CHR}}.peak_motif.bed" -o "${{OUTPUT_BASE}}.${{CHR}}.peak_motif.bed"
        bedtools groupby -i "${{OUTPUT_BASE}}.${{CHR}}.peak_motif.bed" -g 1-4 -c 5 -o sum > "${{OUTPUT_BASE}}.${{CHR}}.peak_motif.bed.tmp"
        mv "${{OUTPUT_BASE}}.${{CHR}}.peak_motif.bed.tmp" "${{OUTPUT_BASE}}.${{CHR}}.peak_motif.bed"
        sort -k1,1V -k2,2n -k3,3n -k4,4 "${{OUTPUT_BASE}}.${{CHR}}.peak_motif.bed" -o "${{OUTPUT_BASE}}.${{CHR}}.peak_motif.bed"
    done

    cat $(ls "${{OUTPUT_BASE}}."*".peak_motif.bed" | grep -v random | grep -v chrY | sort -k1,1V) > "$OUTPUT_PATH"

    # Clean up temporary files
    rm "${{OUTPUT_BASE}}."*".peak.bed" "${{OUTPUT_BASE}}."*".motif.bed" "${{OUTPUT_BASE}}."*".peak_motif.bed"

    echo "Peak motif extraction completed. Results saved in $OUTPUT_PATH"
    """
    subprocess.run(cmd, shell=True, check=True)
    return "get_motif.bed"


def create_peak_motif(peak_motif_bed, output_zarr, peak_bed):
    """
    Create a peak motif zarr file from a peak motif bed file.

    This function reads a peak motif bed file, pivots the data, and saves it to a zarr file.
    The zarr file contains three datasets: 'data', 'peak_names', 'motif_names', and 'accessibility'.
    The 'data' dataset is a sparse matrix containing the peak motif data.
    The 'peak_names' dataset contains the peak names.
    The 'motif_names' dataset contains the motif names.

    Args:
        peak_motif_bed (str): Path to the peak motif bed file.
        output_zarr (str): Path to the output zarr file.
    """
    # Read the peak motif bed file
    peak_motif = pd.read_csv(
        peak_motif_bed,
        sep="\t",
        header=None,
        names=["Chromosome", "Start", "End", "Motif_cluster", "Score"],
    )

    # Pivot the data
    peak_motif_pivoted = peak_motif.pivot_table(
        index=["Chromosome", "Start", "End"],
        columns="Motif_cluster",
        values="Score",
        fill_value=0,
    )
    peak_motif_pivoted.reset_index(inplace=True)

    # Create the 'Name' column
    peak_motif_pivoted["Name"] = peak_motif_pivoted.apply(
        lambda x: f'{x["Chromosome"]}:{x["Start"]}-{x["End"]}', axis=1
    )
    peak_motif_pivoted = peak_motif_pivoted.drop(columns=["Chromosome", "Start", "End"])
    # Read the original peak bed file
    original_peaks = pd.read_csv(
        peak_bed, sep="\t", header=None, names=["Chromosome", "Start", "End", "Score"]
    )
    # exclude chrM and chrY
    original_peaks = original_peaks[~original_peaks.Chromosome.isin(["chrM", "chrY"])]
    original_peaks["Name"] = original_peaks.apply(
        lambda x: f'{x["Chromosome"]}:{x["Start"]}-{x["End"]}', axis=1
    )

    # Merge the pivoted data with the original peaks
    merged_data = pd.merge(original_peaks, peak_motif_pivoted, on="Name", how="left")

    # Fill NaN values with 0 for motif columns
    motif_columns = [
        col
        for col in merged_data.columns
        if col not in ["Chromosome", "Start", "End", "Score", "Name"]
    ]
    merged_data[motif_columns] = merged_data[motif_columns].fillna(0)
    # Prepare data for zarr storage
    name_values = list(merged_data["Name"].values)
    motif_values = motif_columns

    # Create sparse matrix
    motif_data_matrix = merged_data[motif_columns].values
    # Open zarr store and save data
    from numcodecs import Blosc

    z = zarr.open(output_zarr, mode="w")
    z.create_dataset(
        "data",
        data=motif_data_matrix.data,
        chunks=(1000, motif_data_matrix.shape[1]),
        dtype=np.float32,
        compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
        shape=motif_data_matrix.shape,
    )
    z.create_dataset("peak_names", data=name_values)
    z.create_dataset("motif_names", data=motif_values)

    print(f"Peak motif data saved to {output_zarr}")


def zip_zarr(zarr_file):
    subprocess.run(["zip", "-r", f"{zarr_file}.zip", zarr_file], check=True)

def unzip_zarr(zarr_file):
    subprocess.run(["unzip", f"{zarr_file}.zip"], check=True)

def add_atpm(zarr_file, bed_file, celltype):
    """
    Add aTPM (ATAC-seq 'Transcript'/Count Per Million) data for a specific cell type to the zarr file.

    This function reads aTPM data from a BED file and adds it to the zarr file under the 'atpm' group.
    The aTPM values are associated with peak names in the zarr file.

    Args:
        zarr_file (str): Path to the zarr file.
        bed_file (str): Path to the BED file containing aTPM data.
        celltype (str): Name of the cell type for which the aTPM data is being added.

    The BED file should have the following columns:
    1. Chromosome
    2. Start
    3. End
    4. aTPM value

    The function creates an 'atpm' group in the zarr file if it doesn't exist and adds a dataset
    for the specified cell type under this group.
    """
    df = pd.read_csv(
        bed_file, sep="\t", header=None, names=["Chromosome", "Start", "End", "aTPM"]
    )
    df["Name"] = df.apply(
        lambda x: f'{x["Chromosome"]}:{x["Start"]}-{x["End"]}', axis=1
    )
    name_to_atpm = dict(zip(df["Name"], df["aTPM"]))
    z = zarr.open(zarr_file, mode="a")

    # Create the atpm group if it doesn't exist
    if "atpm" not in z:
        z.create_group("atpm")

    # Save aTPM data for the specific cell type
    z["atpm"].create_dataset(
        celltype,
        data=np.array([name_to_atpm[name] for name in z["peak_names"]]),
        overwrite=True,
        chunks=(1000,),
        dtype=np.float32,
    )


def add_exp(
    zarr_file, rna_file, atac_file, celltype, assembly="hg38", version=40, extend_bp=100
):
    """
    Add expression and TSS data for a specific cell type to the zarr file.
    """
    # Initialize Gencode
    gencode = Gencode(assembly=assembly, version=version)

    # Read RNA data
    gene_exp = pd.read_csv(rna_file)
    gene_exp["gene_id"] = gene_exp["gene_id"].apply(lambda x: x.split(".")[0])
    promoter_exp = pd.merge(
        gencode.gtf, gene_exp, left_on="gene_id", right_on="gene_id"
    )

    # Read ATAC data
    if atac_file.endswith(".bed"):
        atac = pr(
            pd.read_csv(
                atac_file,
                sep="\t",
                header=None,
                names=["Chromosome", "Start", "End", "aTPM"],
            ).reset_index(),
            int64=True,
        )
    else:
        atac = pr(pd.read_csv(atac_file, index_col=0).reset_index(), int64=True)

    # Join ATAC and RNA data
    exp = atac.join(pr(promoter_exp, int64=True).extend(extend_bp), how="left").as_df()
    
    # Save to exp.feather for getting gene name to index
    gene_idx_info = exp.query('index_b!=-1')[['index', 'gene_name', 'Strand']].values

    
    # Process expression data
    exp = (
        exp[["index", "Strand", "TPM"]]
        .groupby(["index", "Strand"])
        .mean()
        .reset_index()
    )

    # Calculate expression and TSS
    exp_n = exp[exp.Strand == "-"].set_index("index")["TPM"].fillna(0)
    exp_p = exp[exp.Strand == "+"].set_index("index")["TPM"].fillna(0)
    exp_n[exp_n < 0] = 0
    exp_p[exp_p < 0] = 0

    exp_n_tss = (exp[exp.Strand == "-"].set_index("index")["TPM"] >= 0).fillna(False)
    exp_p_tss = (exp[exp.Strand == "+"].set_index("index")["TPM"] >= 0).fillna(False)

    tss = np.stack([exp_p_tss, exp_n_tss]).T
    exp_data = np.stack([exp_p, exp_n]).T

    # Open zarr file
    z = zarr.open(zarr_file, mode="a")

    # Create groups if they don't exist
    for group in ["expression_positive", "expression_negative", "tss"]:
        if group not in z:
            z.create_group(group)

    # Save data for the specific cell type
    peak_names = z["peak_names"][:]
    z["expression_positive"].create_dataset(
        celltype,
        data=exp_data[:, 0].astype(np.float32),
        overwrite=True,
        chunks=(1000,),
        dtype=np.float32,
    )
    z["expression_negative"].create_dataset(
        celltype,
        data=exp_data[:, 1].astype(np.float32),
        overwrite=True,
        chunks=(1000,),
        dtype=np.float32,
    )
    z["tss"].create_dataset(
        celltype,
        data=tss.astype(np.int8),
        overwrite=True,
        chunks=(1000,),
        dtype=np.int8,
    )
    z["gene_idx_info_index"] = gene_idx_info[:, 0].astype(int)
    z["gene_idx_info_name"] = gene_idx_info[:, 1].astype(str)
    z["gene_idx_info_strand"] = gene_idx_info[:, 2].astype(str)
