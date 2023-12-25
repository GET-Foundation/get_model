#%%
from random import sample
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
    
    def sample_boundary(self, chromosome, start, end, window_size=10, threshold_factor=0.1):
        """
        Samples a boundary from a given chromosome.

        :param chromosome: Chromosome identifier (e.g., 'chr1').
        :param window_size: Window size for calculating the moving average.
        :return: Tuple of numpy arrays (positions, insulation scores).
        """
        positions, insulation = self.calculate_insulation(chromosome, start, end, window_size)
        local_minima = self.find_local_minima(insulation, threshold_factor)
        local_minima_pos = [start] + list(positions[local_minima]) 
        # all possible pairs of local_minima_pos
        samples = np.array(np.meshgrid(local_minima_pos, local_minima_pos)).T.reshape(-1,2)
        # add consecutive local minima as samples
        final_samples = []
        for i in samples:
            if i[0] > i[1] and i[0] - i[1] > 5000 and i[0] - i[1] < 3000000:
                final_samples.append((i[1], i[0]))
            elif i[1] - i[0] > 5000 and i[1] - i[0] < 3000000:
                final_samples.append((i[0], i[1]))
        return np.array(list(set(final_samples)))
    
        
    def adjacent_boundary(self, chromosome, start, end, window_size=2, threshold_factor=0):
        """
        Samples a boundary from a given chromosome.

        :param chromosome: Chromosome identifier (e.g., 'chr1').
        :param window_size: Window size for calculating the moving average.
        :return: Tuple of numpy arrays (positions, insulation scores).
        """
        positions, insulation = self.calculate_insulation(chromosome, start, end, window_size)
        local_minima = self.find_local_minima(insulation, threshold_factor)
        local_minima_pos = [start] + list(positions[local_minima]) 
        # all possible pairs of local_minima_pos
        samples = np.stack([local_minima_pos[:-1], local_minima_pos[1:]], axis=1)
        # add consecutive local minima as samples
        final_samples = []
        for i in samples:
            if i[0] > i[1] and i[0] - i[1] > 1000 and i[0] - i[1] < 200000:
                final_samples.append((i[1], i[0]))
            elif i[1] - i[0] > 1000 and i[1] - i[0] < 200000:
                final_samples.append((i[0], i[1]))
        return np.array(list(set(final_samples)))
    
    def sample_boundary_with_ctcf(self, chromosome, start, end, window_size=10, threshold_factor=0.1):
        """
        Samples a boundary from a given chromosome.

        :param chromosome: Chromosome identifier (e.g., 'chr1').
        :param window_size: Window size for calculating the moving average.
        :return: Tuple of numpy arrays (positions, insulation scores).
        """
        sample_list = self.sample_boundary(chromosome, start, end, window_size, threshold_factor)
        ctcf_df = self.ctcf_df.query(f'Chromosome=="{chromosome}" and Start>={start} and End<={end}')
        result = []
        for c_start, c_end in sample_list:
            ctcf_start = ctcf_df.query('Start>@c_start-15000 & Start<@c_start+15000')
            ctcf_end = ctcf_df.query('End>@c_end-15000 & End<@c_end+15000')
            if len(ctcf_start) == 0 or len(ctcf_end) == 0:
                continue
            else:
                ctcf_start = ctcf_start.sample(1, weights='num_celltype')
                ctcf_end = ctcf_end.sample(1, weights='num_celltype')
                result.append((ctcf_start.Start.values[0], ctcf_end.End.values[0], (ctcf_start.num_celltype.values[0]+ctcf_end.num_celltype.values[0])/2))
        result = pd.DataFrame(result, columns=['Start', 'End', 'mean_num_celltype'])
        result['Chromosome'] = chromosome
        result['Distance'] = result.End - result.Start
        return result.query('Distance>5000')
    
    def adjacent_boundary_with_ctcf(self, chromosome, start, end, window_size=2, threshold_factor=0):
        """
        Samples a boundary from a given chromosome.

        :param chromosome: Chromosome identifier (e.g., 'chr1').
        :param window_size: Window size for calculating the moving average.
        :return: Tuple of numpy arrays (positions, insulation scores).
        """
        sample_list = self.adjacent_boundary(chromosome, start, end, window_size, threshold_factor)
        ctcf_df = self.ctcf_df.query(f'Chromosome=="{chromosome}" and Start>={start} and End<={end}')
        result = []
        for c_start, c_end in sample_list:
            ctcf_start = ctcf_df.query('Start>@c_start-2500 & Start<@c_start+2500')
            ctcf_end = ctcf_df.query('End>@c_end-2500 & End<@c_end+2500')
            if len(ctcf_start) == 0 or len(ctcf_end) == 0:
                result.append((c_start, c_end, 0))
            else:
                ctcf_start = ctcf_start.sample(1, weights='num_celltype')
                ctcf_end = ctcf_end.sample(1, weights='num_celltype')
                result.append((ctcf_start.Start.values[0], ctcf_end.End.values[0], (ctcf_start.num_celltype.values[0]+ctcf_end.num_celltype.values[0])/2))
        result = pd.DataFrame(result, columns=['Start', 'End', 'mean_num_celltype'])
        result['Chromosome'] = chromosome
        result['Distance'] = result.End - result.Start
        return result.query('Distance>1000')
    
    def sample_for_chromosome(self, chromosome, chunk_size = 4000000, window_size=10, threshold_factor=0.1):
        """
        Samples a boundary from a given chromosome.

        :param chromosome: Chromosome identifier (e.g., 'chr1').
        :param window_size: Window size for calculating the moving average.
        :return: Tuple of numpy arrays (positions, insulation scores).
        """
        results = []
        chrom_size = self.bw.chroms()[chromosome]
        for start in range(0, chrom_size, 1000000):
            try:
                l = self.sample_boundary_with_ctcf(chromosome, start, start + chunk_size, window_size=window_size, threshold_factor=threshold_factor)
                results.append(l)
            except:
                continue
        results = pd.concat(results)
        return results
    
    def sample_for_chromosome_adjacent(self, chromosome, chunk_size = 2000000, window_size=2, threshold_factor=0):
        """
        Samples a boundary from a given chromosome.

        :param chromosome: Chromosome identifier (e.g., 'chr1').
        :param window_size: Window size for calculating the moving average.
        :return: Tuple of numpy arrays (positions, insulation scores).
        """
        results = []
        chrom_size = self.bw.chroms()[chromosome]
        for start in range(0, chrom_size, 1000000):
            try:
                l = self.adjacent_boundary_with_ctcf(chromosome, start, start + chunk_size, window_size=window_size, threshold_factor=threshold_factor)
                results.append(l)
            except:
                continue
        results = pd.concat(results)
        return results.drop_duplicates()
#%%
# Example usage
insulation = InsulationSampler('../../data/4DN_average_insulation.bw', '../../data/hg38.ctcf_motif_count.num_celltype_gt_5.feather')
insulation.adjacent_boundary_with_ctcf('chr1', 0, 4000000)
#%%
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
with ThreadPoolExecutor(max_workers=96) as executor:
    results = tqdm(executor.map(insulation.sample_for_chromosome, insulation.get_chroms().keys()))

results = pd.concat(results)
#%%
results.Distance.hist(bins=100)
#%%
results = results.drop_duplicates()
#%%
results.Distance.hist(bins=100, log=True)
#%%
results.to_feather('../../data/hg38_4DN_average_insulation.ctcf.longrange.feather')


#%%
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
results_adjecent = []
for threshold in [0, 0.2, 0.4, 0.6, 0.8, 1]:
    for window in [1, 2, 5]:
        with ThreadPoolExecutor(max_workers=96) as executor:
            result_adjecent = tqdm(executor.map(insulation.sample_for_chromosome_adjacent, insulation.get_chroms().keys(), [2000000]*24,[window]*24,[threshold]*24))
        result_adjecent = pd.concat(result_adjecent)
        result_adjecent['window'] = window
        result_adjecent['threshold'] = threshold
        results_adjecent.append(result_adjecent)
#%%
results_adjecent = pd.concat(results_adjecent).drop_duplicates()
#%%
results_adjecent.to_feather('../../data/hg38_4DN_average_insulation.ctcf.adjecent.feather')
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
