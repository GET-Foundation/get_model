#%%
import pyBigWig
import numpy as np
import os
import os.path
import numpy as np
import pandas as pd
from caesar.io.genome import ChromSize
class InsulationSampler:
    """
    A class for processing BigWig files to analyze chromosomal data.
    """

    def __init__(self, file_path, ctcf_feather=None):
        """
        Initializes the InsulationSampler with a BigWig file.

        :param file_path: Path to the BigWig file.
        """
        self.file_path = file_path
        self.bw = pyBigWig.open(file_path)
        if ctcf_feather is not None:
            self.ctcf_df = pd.read_feather(ctcf_feather)

    def get_chroms(self):
        """
        Returns the chromosome information from the BigWig file.
        """
        return self.bw.chroms()

    def calculate_insulation(self, chromosome, start, end, window_size=10):
        """
        Calculates the insulation score for a given chromosome.

        :param chromosome: Chromosome identifier (e.g., 'chr1').
        :param window_size: Window size for calculating the moving average.
        :return: Tuple of numpy arrays (positions, insulation scores).
        """
        chrom_size = self.bw.chroms()[chromosome]
        if end > chrom_size:
            end = chrom_size
        intervals = self.bw.intervals(chromosome, start, end)
        insulation = np.array([i[2] for i in intervals])
        positions = np.array([i[1] for i in intervals])

        # Apply convolution to smooth the insulation scores
        insulation_smooth = np.convolve(insulation, np.ones(window_size)/window_size, 'same')

        return positions, insulation_smooth

    def find_local_minima(self, insulation, threshold_factor=0.1):
        """
        Identifies local minima in the insulation score array.

        :param insulation: Insulation score array.
        :param threshold_factor: Factor to adjust the threshold for local minima.
        :return: Array of positions of local minima.
        """
        local_min = np.r_[True, insulation[1:] < insulation[:-1]] & np.r_[insulation[:-1] < insulation[1:], True]
        mean_threshold = insulation.mean() - threshold_factor * insulation.std()
        local_min = local_min & (insulation < mean_threshold)

        return np.where(local_min)
    
    def sample_boundary(self, chromosome, start, end, n=10, window_size=10, threshold_factor=0.1):
        """
        Samples a boundary from a given chromosome.

        :param chromosome: Chromosome identifier (e.g., 'chr1').
        :param window_size: Window size for calculating the moving average.
        :return: Tuple of numpy arrays (positions, insulation scores).
        """
        positions, insulation = self.calculate_insulation(chromosome, start, end, window_size)
        local_minima = self.find_local_minima(insulation, threshold_factor)
        samples = np.random.choice(positions[local_minima], (n, 2)) 
        # add consecutive local minima as samples
        samples = np.concatenate([samples, np.stack((positions[local_minima][0:-1], positions[local_minima][1:]), axis=1)])
        final_samples = []
        for i in samples:
            if i[0] > i[1] and i[0] - i[1] > 10000 and i[0] - i[1] < 3000000:
                final_samples.append((i[1], i[0]))
            elif i[1] - i[0] > 10000 and i[1] - i[0] < 3000000:
                final_samples.append((i[0], i[1]))
        return np.array(list(set(final_samples))).T
    
    def sample_boundary_with_ctcf(self, chromosome, start, end, n=10, window_size=10, threshold_factor=0.1):
        """
        Samples a boundary from a given chromosome.

        :param chromosome: Chromosome identifier (e.g., 'chr1').
        :param window_size: Window size for calculating the moving average.
        :return: Tuple of numpy arrays (positions, insulation scores).
        """
        sample_list = self.sample_boundary(chromosome, start, end, n, window_size, threshold_factor)
        sample_list_min = sample_list.min()
        sample_list_max = sample_list.max()
        ctcf_df = self.ctcf_df.query(f'Chromosome=="{chromosome}" and Start>={sample_list_min} and End<={sample_list_max}')
        ctcf_df = ctcf_df.query('num_celltype>20')
        result = [(ctcf_df.query('Start>@start & End<@end').Start.values[0], ctcf_df.query('Start>@start-5000 & End<@end+5000').End.values[-1]) for (start, end) in sample_list.T] 
        result = pd.DataFrame(result, columns=['start', 'end'])
        result['distance'] = result.end - result.start
        return result.query('distance>10000')
    
    def sample_for_chromosome(self, chromosome, chunk_size = 4000000, window_size=10, threshold_factor=0.1):
        """
        Samples a boundary from a given chromosome.

        :param chromosome: Chromosome identifier (e.g., 'chr1').
        :param window_size: Window size for calculating the moving average.
        :return: Tuple of numpy arrays (positions, insulation scores).
        """
        results = []
        results_seed = []
        for seed in range(500):
            chrom_size = self.bw.chroms()[chromosome]
            for i in range(chrom_size//chunk_size):
                try:
                    l = self.sample_boundary_with_ctcf(chromosome, i*chunk_size, (i+1)*chunk_size, n=chunk_size//100000, window_size=window_size, threshold_factor=threshold_factor)
                    results_seed.append(l)
                except:
                    continue
            results_seed = pd.concat(results_seed)
            results_seed['seed'] = seed
            results.append(results_seed)
        results = pd.concat(results)
        results['chromosome'] = chromosome
        return results
#%%
# Example usage
insulation = InsulationSampler('../../data/4DN_average_insulation.bw', '../../data/hg38.ctcf_motif_count.num_celltype_gt_5.feather')
#%%
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
with ThreadPoolExecutor(max_workers=23) as executor:
    results = tqdm(executor.map(insulation.sample_for_chromosome, insulation.get_chroms().keys()))

results = pd.concat(results)
#%%
results.distance.hist(bins=100)
#%%
#%%
class CTCFBoundary(object):

    """A class to read and generate TAD boundaries"""
    def __init__(self, data_dir, assembly) -> None:
        self.df = pd.read_feather(os.path.join(data_dir, f'{assembly}.ctcf_motif_count.num_celltype_gt_5.feather'))
        self.chrom_size = ChromSize(assembly, data_dir).dict
    
    def get_boundary_list(self, fun='conservation'):
        if fun == 'conservation':
            # call get_boundary_by_celltype_conservation
            self.boundary_list = self.get_boundary_by_celltype_conservation()
        return 
    
    def draw_from_boundary_list(self, window_size=2_000_000):
            """
            Randomly draws intervals from the boundary list for each chromosome.

            Args:
                window_size (int): The size of the window for each interval.

            Returns:
                dict: A dictionary containing the chromosome names as keys and a list of intervals as values.
            """
            boundary_dictionary = {}
            for chr_name, starts in tqdm(self.boundary_list.items()):
                interval_list = []
                for i in range(self.chrom_size[chr_name]//window_size):
                    random_number = np.random.randint(0, self.chrom_size[chr_name]-window_size)
                    raw_interval = (random_number, random_number+window_size)
                    adjusted_interval = self.get_nearest_boundary(raw_interval, starts)
                    if adjusted_interval[1] - adjusted_interval[0] < 1_000_000:
                        continue
                    interval_list.append(adjusted_interval)
                boundary_dictionary[chr_name] = interval_list
            return boundary_dictionary

    def get_nearest_boundary(self, raw_interval, starts):
        """
        Given a raw_interval and a list of starts, find the nearest boundary to the raw_interval
        """
        start = raw_interval[0]
        end = raw_interval[1]
        nearest_start = min(starts, key=lambda x:abs(x-start))
        nearest_end = min(starts, key=lambda x:abs(x-end))
        return (nearest_start, nearest_end)        
    
    def get_boundary_by_celltype_conservation(self):
        filtered_df = self.df.query('num_celltype>100')
        # dict with chr as key and starts as list value
        filtered_df.set_index('Chromosome')
        boundary_list = {}
        for chr_name, group in filtered_df.groupby('Chromosome'):
            starts = group['Start'].tolist()
            if len(starts) == 0:
                continue
            starts  = [0] + starts + [self.chrom_size[chr_name]]
            boundary_list[chr_name] = starts
        return boundary_list

    def get_boundary_from_tad_list(self):
        """
        Read TAD list from self.data_dir/TADs/{assembly}/*
        All files are in bed format, with 3 columns: chr, start, end
        some of the chr are in format chr1, some are in format 1, convert them all to have 'chr'
        """
        boundary_list = {}
        for chr_name in self.chrom_size.keys():
            if chr_name.startswith('chr'):
                chr_name = chr_name[3:]
            else:
                chr_name = 'chr' + chr_name
            boundary_list[chr_name] = []
        for file in os.listdir(os.path.join(self.data_dir, 'TADs', self.assembly)):
            chr_name = file.split('.')[0]
            if chr_name.startswith('chr'):
                chr_name = chr_name[3:]
            else:
                chr_name = 'chr' + chr_name
            df = pd.read_csv(os.path.join(self.data_dir, 'TADs', self.assembly, file), sep='\t', header=None)
            starts = df[1].tolist()
            boundary_list[chr_name] = starts
        
        return boundary_list

        
# # %%
# import os
# import os.path
# from typing import Any, Callable, Optional, Tuple, List, Dict
# from tqdm import tqdm
# import numpy as np
# import pandas as pd
# from caesar.io.genome import ChromSize
# c = CTCFBoundary('/manitou/pmg/users/xf2217/get_model/data/', 'hg38')
# # %%
# c.get_boundary_list()
# # %%
# c.draw_from_boundary_list()
# # %%
