from pyranges import read_gtf
from pyranges import PyRanges as pr
import pandas as pd
from tqdm import tqdm
import numpy as np
import os

class Gencode(object):
    """Read gencode gene annotation given genome assembly and version, 
    returns a pandas dataframe"""

    def __init__(self, assembly="hg38", version=40, gtf_dir=".", exclude_chrs=['chrM', 'chrY']):
        super(Gencode, self).__init__()

        self.assembly = assembly
        self.gtf_dir = gtf_dir  # New parameter for specifying GTF file location

        if self.assembly == "hg38":
            self.url = "http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{version}/gencode.v{version}.basic.annotation.gtf.gz".format(
                version=str(version))
            self.gtf = os.path.join(self.gtf_dir, "gencode.v{version}.basic.annotation.gtf.gz".format(
                version=str(version)))
        elif self.assembly == "mm10":
            self.url = "http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_{version}/gencode.v{version}.basic.annotation.gtf.gz".format(
                version=str(version))
            self.gtf = os.path.join(self.gtf_dir, "gencode.{assembly}.v{version}.basic.annotation.gtf.gz".format(
                assembly=self.assembly, version=str(version)))
        elif self.assembly == "hg19":
            self.url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{version}/GRCh37_mapping/gencode.v{version}lift37.basic.annotation.gtf.gz".format(
                version=str(version))
            self.gtf = os.path.join(self.gtf_dir, "gencode.v{version}lift37.basic.annotation.gtf.gz".format(
                version=str(version)))

        if os.path.exists(os.path.join(self.gtf_dir, "gencode.v{version}.{assembly}.feather".format(version=str(version), assembly=self.assembly))):
            self.gtf = pd.read_feather(os.path.join(self.gtf_dir, "gencode.v{version}.{assembly}.feather".format(
                version=str(version), assembly=self.assembly)))
            self.feather_file = os.path.join(self.gtf_dir, "gencode.v{version}.{assembly}.feather".format(
                version=str(version), assembly=self.assembly))
        else:
            if os.path.exists(self.gtf):
                self.gtf = read_gtf(self.gtf).as_df()
            else:
                # download gtf to the specified directory
                os.system(
                    "wget -P {dir} {url} -O {gtf}".format(dir=self.gtf_dir, url=self.url, gtf=self.gtf))
                self.gtf = read_gtf(self.gtf).as_df()

            positive = self.gtf[(self.gtf.Feature == 'transcript') & (
                self.gtf.Strand == '+')][['Chromosome', 'Start', 'Start', 'Strand', 'gene_name', 'gene_id', 'gene_type']]
            negative = self.gtf[(self.gtf.Feature == 'transcript') & (
                self.gtf.Strand == '-')][['Chromosome', 'End', 'End', 'Strand', 'gene_name', 'gene_id', 'gene_type']]

            positive.columns = ['Chromosome', 'Start',
                                'End', 'Strand', 'gene_name', 'gene_id', 'gene_type']
            negative.columns = ['Chromosome', 'Start',
                                'End', 'Strand', 'gene_name', 'gene_id', 'gene_type']

            self.gtf = pd.concat([positive, negative],
                                 axis=0).drop_duplicates().reset_index()
            self.gtf['gene_id'] = self.gtf.gene_id.str.split(".").str[0]
            self.gtf.to_feather(os.path.join(self.gtf_dir, "gencode.v{version}.{assembly}.feather".format(
                version=str(version), assembly=self.assembly)))
            self.feather_file = os.path.join(self.gtf_dir, "gencode.v{version}.{assembly}.feather".format(
                version=str(version), assembly=self.assembly))
        self.gtf = self.gtf[~self.gtf.Chromosome.isin(exclude_chrs)]
        # count number of chromosomes per gene and remove the genes with multiple chromosomes
        self.gtf['chrom_count'] = self.gtf.groupby('gene_name')['Chromosome'].transform(
            'nunique')
        self.gtf = self.gtf[self.gtf.chrom_count == 1]
        return


    def get_exp_feather(self, peaks, extend_bp=300):
        exp = pr(peaks, int64=True).join(pr(self.gtf, int64=True).extend(
            extend_bp), how='left').as_df()
        return exp.reset_index(drop=True)

    def query_region(self, chrom, start, end, strand=None):
        result = self.gtf.query('Chromosome == "{chrom}" & Start > {start} & End < {end}'.format(chrom=chrom, start=start, end=end))
        if strand is not None:
            result = result.query('Strand == "{strand}"'.format(strand=strand))
        return result
    
