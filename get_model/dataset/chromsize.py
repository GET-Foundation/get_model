from io import StringIO
import pandas as pd
import gzip
import requests
import os


class ChromSize:
    def __init__(self, assembly=None, annotation_dir=None):
        self.assembly = assembly
        self.annotation_dir = annotation_dir
        if self.assembly is None:
            raise ValueError("assembly is not specified")
        if self.annotation_dir is None:
            raise ValueError("annotation_dir is not specified")

        self.chrom_sizes = self.parse_or_download_chrom_sizes()

    def _download_chrom_sizes(self):
        url = f"http://hgdownload.soe.ucsc.edu/goldenPath/{self.assembly}/bigZips/{self.assembly}.chrom.sizes"
        response = requests.get(url)
        if response.status_code != 200:
            raise ConnectionError("Failed to download chromosome data")
        return self._parse_chrom_data(response.text)

    def _parse_chrom_data(self, data):
        chrom_sizes = {}
        lines = data.strip().split('\n')
        for line in lines:
            parts = line.split('\t')
            if len(parts) == 2:
                chrom, length = parts
                chrom_sizes[chrom] = int(length)
        return chrom_sizes

    def get_dict(self, chr_included=None):
        if chr_included is None:
            return self.chrom_sizes
        else:
            return {chr: self.chrom_sizes.get(chr, None) for chr in chr_included}

    # property as a dict
    @property
    def dict(self):
        return self.chrom_sizes

    def save_chrom_sizes(self):
        filepath = os.path.join(self.annotation_dir,
                                f"{self.assembly}_chrom_sizes.txt")
        with open(filepath, 'w') as file:
            for chrom, length in self.chrom_sizes.items():
                file.write(f"{chrom}\t{length}\n")

    def parse_or_download_chrom_sizes(self):
        filepath = os.path.join(self.annotation_dir,
                                f"{self.assembly}_chrom_sizes.txt")
        if os.path.exists(filepath):
            with open(filepath, 'r') as data:
                return self._parse_chrom_data(data.read())
        else:
            return self._download_chrom_sizes()

    def as_pyranges(self):
        try:
            import pyranges as pr
            cs = pd.DataFrame({'Chromosome': list(self.chrom_sizes.keys()), 'Start': 0, 'End': list(self.chrom_sizes.values())}).sort_values(by=['Chromosome', 'Start', 'End'])
            return pr.PyRanges(cs, int64=True)
        except ImportError:
            raise ImportError("pyranges is not installed")

    def __repr__(self) -> str:
        return f"ChromSize(assembly={self.assembly}, annotation_dir={self.annotation_dir})"


class ChromGap:
    """
    A class to download, parse, and analyze AGP (A Golden Path) files from UCSC.

    This class provides functionality to automatically download and parse AGP files,
    extract telomere information, and work with chromosome gap data.

    Attributes:
        assembly (str): The genome assembly name (e.g., "hg38").
        annotation_dir (str): The directory to store downloaded and parsed files.
        agp_data (pd.DataFrame): The parsed AGP data as a pandas DataFrame.
    """

    def __init__(self, assembly=None, annotation_dir=None, config: Config = None):
        """
        Initialize the ChromGap object.

        Args:
            assembly (str, optional): The genome assembly name.
            annotation_dir (str, optional): The directory to store files.
            config (Config, optional): A Config object containing assembly and annotation_dir.

        Raises:
            ValueError: If assembly or annotation_dir is not specified.
        """
        if config is not None:
            self.assembly = config.get("assembly")
            self.annotation_dir = config.get("annotation_dir")
        else:
            self.assembly = assembly
            self.annotation_dir = annotation_dir

        if self.assembly is None:
            raise ValueError("assembly is not specified")
        if self.annotation_dir is None:
            raise ValueError("annotation_dir is not specified")

        self.agp_data = self.parse_or_download_agp()

    def _download_agp(self):
        """
        Download the AGP file from UCSC.

        Returns:
            str: The content of the downloaded AGP file.

        Raises:
            ConnectionError: If the download fails.
        """
        url = f"https://hgdownload.soe.ucsc.edu/goldenPath/{self.assembly}/bigZips/{self.assembly}.agp.gz"
        response = requests.get(url)
        if response.status_code != 200:
            raise ConnectionError("Failed to download AGP data")
        return gzip.decompress(response.content).decode('utf-8')

    def _parse_agp_data(self, data):
        """
        Parse the AGP data into a pandas DataFrame.

        Args:
            data (str): The AGP file content.

        Returns:
            pd.DataFrame: The parsed AGP data.
        """
        columns = ['chrom', 'start', 'end', 'part_number', 'component_type',
                   'component_id', 'component_start', 'component_end', 'orientation']
        return pd.read_csv(StringIO(data), sep='\t', comment='#', names=columns)

    def get_telomeres(self, return_tabix=False):
        """
        Extract telomere information from the AGP data.

        Returns:
            pd.DataFrame: A DataFrame containing telomere information.
        """
        df = self.agp_data[self.agp_data['component_start'] == 'telomere']
        if return_tabix:
            return pandas_to_tabix_region(df)
        return df

    def get_heterochromatin(self, return_tabix=False):
        """
        Extract heterochromatin information from the AGP data.

        Returns:
            pd.DataFrame: A DataFrame containing heterochromatin information.
        """
        df = self.agp_data[self.agp_data['component_start'].isin(['heterochromatin', 'centromere'])]
        if return_tabix:
            return pandas_to_tabix_region(df)
        return df

    def save_agp_data(self):
        """
        Save the AGP data to a file in the annotation directory.
        """
        filepath = os.path.join(self.annotation_dir,
                                f"{self.assembly}_agp.txt")
        self.agp_data.to_csv(filepath, sep='\t', index=False)

    def parse_or_download_agp(self):
        """
        Parse existing AGP file or download and parse if not available.

        Returns:
            pd.DataFrame: The parsed AGP data.
        """
        filepath = os.path.join(self.annotation_dir,
                                f"{self.assembly}_agp.txt")
        if os.path.exists(filepath):
            return pd.read_csv(filepath, sep='\t')
        else:
            data = self._download_agp()
            agp_data = self._parse_agp_data(data)
            self.agp_data = agp_data
            self.save_agp_data()
            return agp_data

    def __repr__(self) -> str:
        """
        Return a string representation of the ChromGap object.

        Returns:
            str: A string representation of the ChromGap object.
        """
        return f"ChromGap(assembly={self.assembly}, annotation_dir={self.annotation_dir})"

def pandas_to_tabix_region(df, chrom_col='chrom', start_col='start', end_col='end'):
    """
    Convert a pandas DataFrame to a tabix region string.

    Args:
        df (pd.DataFrame): The DataFrame to convert.
        chrom_col (str): The column name for the chromosome.
        start_col (str): The column name for the start position.
        end_col (str): The column name for the end position.

    Returns:
        str: A tabix region string.
    """
    return ' '.join(df.apply(lambda x: f"{x[chrom_col]}:{x[start_col]}-{x[end_col]}", axis=1).tolist())