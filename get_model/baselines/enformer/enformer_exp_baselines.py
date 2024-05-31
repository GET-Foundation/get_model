# %%
import tensorflow as tf
# %%
tf.config.list_physical_devices('GPU')
# %%
import tensorflow_hub as hub
import joblib
import gzip
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm


transform_path = 'gs://dm-enformer/models/enformer.finetuned.SAD.robustscaler-PCA500-robustscaler.transform.pkl'
model_path = "/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/ckpts/enformer-base"
fasta_file = '/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/data/genome.fa'
clinvar_vcf = '/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/data/clinvar.vcf.gz'
targets_txt = "/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/data/targets_human.txt"
output_dir = "/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/preds"


SEQUENCE_LENGTH = 393216
OUTPUT_SEQ_LENGTH = 114688
BATCH_SIZE = 12
SAVE_EVERY_N_BATCHES = 100


class Enformer:

  def __init__(self, tfhub_url):
    self._model = hub.load(tfhub_url).model

  def predict_on_batch(self, inputs):
    predictions = self._model.predict_on_batch(inputs)
    return {k: v.numpy() for k, v in predictions.items()}

  @tf.function
  def contribution_input_grad(self, input_sequence,
                              target_mask, output_head='human'):
    input_sequence = input_sequence[tf.newaxis]

    target_mask_mass = tf.reduce_sum(target_mask)
    with tf.GradientTape() as tape:
      tape.watch(input_sequence)
      prediction = tf.reduce_sum(
          target_mask[tf.newaxis] *
          self._model.predict_on_batch(input_sequence)[output_head]) / target_mask_mass

    input_grad = tape.gradient(prediction, input_sequence) * input_sequence
    input_grad = tf.squeeze(input_grad, axis=0)
    return tf.reduce_sum(input_grad, axis=-1)


class EnformerScoreVariantsRaw:

  def __init__(self, tfhub_url, organism='human'):
    self._model = Enformer(tfhub_url)
    self._organism = organism
  
  def predict_on_batch(self, inputs):
    ref_prediction = self._model.predict_on_batch(inputs['ref'])[self._organism]
    alt_prediction = self._model.predict_on_batch(inputs['alt'])[self._organism]

    return alt_prediction.mean(axis=1) - ref_prediction.mean(axis=1)


class EnformerScoreVariantsNormalized:

  def __init__(self, tfhub_url, transform_pkl_path,
               organism='human'):
    assert organism == 'human', 'Transforms only compatible with organism=human'
    self._model = EnformerScoreVariantsRaw(tfhub_url, organism)
    with tf.io.gfile.GFile(transform_pkl_path, 'rb') as f:
      transform_pipeline = joblib.load(f)
    self._transform = transform_pipeline.steps[0][1]  # StandardScaler.
    
  def predict_on_batch(self, inputs):
    scores = self._model.predict_on_batch(inputs)
    return self._transform.transform(scores)


class EnformerScoreVariantsPCANormalized:

  def __init__(self, tfhub_url, transform_pkl_path,
               organism='human', num_top_features=500):
    self._model = EnformerScoreVariantsRaw(tfhub_url, organism)
    with tf.io.gfile.GFile(transform_pkl_path, 'rb') as f:
      self._transform = joblib.load(f)
    self._num_top_features = num_top_features
    
  def predict_on_batch(self, inputs):
    scores = self._model.predict_on_batch(inputs)
    return self._transform.transform(scores)[:, :self._num_top_features]


# TODO(avsec): Add feature description: Either PCX, or full names.
# %%
# @title `variant_centered_sequences`

class FastaStringExtractor:
    
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()


def variant_generator(vcf_file, gzipped=False):
  """Yields a kipoiseq.dataclasses.Variant for each row in VCF file."""
  def _open(file):
    return gzip.open(vcf_file, 'rt') if gzipped else open(vcf_file)
    
  with _open(vcf_file) as f:
    for line in f:
      if line.startswith('#'):
        continue
      chrom, pos, id, ref, alt_list = line.split('\t')[:5]
      # Split ALT alleles and return individual variants as output.
      for alt in alt_list.split(','):
        yield kipoiseq.dataclasses.Variant(chrom=chrom, pos=pos,
                                           ref=ref, alt=alt, id=id)


def one_hot_encode(sequence):
  return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


def variant_centered_sequences(vcf_file, sequence_length, gzipped=False,
                               chr_prefix=''):
  seq_extractor = kipoiseq.extractors.VariantSeqExtractor(
    reference_sequence=FastaStringExtractor(fasta_file))

  for variant in variant_generator(vcf_file, gzipped=gzipped):
    interval = Interval(chr_prefix + variant.chrom,
                        variant.pos, variant.pos)
    interval = interval.resize(sequence_length)
    center = interval.center() - interval.start

    reference = seq_extractor.extract(interval, [], anchor=center)
    alternate = seq_extractor.extract(interval, [variant], anchor=center)

    yield {'inputs': {'ref': one_hot_encode(reference),
                      'alt': one_hot_encode(alternate)},
           'metadata': {'chrom': chr_prefix + variant.chrom,
                        'pos': variant.pos,
                        'id': variant.id,
                        'ref': variant.ref,
                        'alt': variant.alt}}

def plot_tracks(tracks, interval, height=1.5):
  fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
  for ax, (title, y) in zip(axes, tracks.items()):
    ax.fill_between(np.linspace(interval.start, interval.end, num=len(y)), y)
    ax.set_title(title)
    sns.despine(top=True, right=True, bottom=True)
  ax.set_xlabel(str(interval))
  plt.tight_layout()


def compute_enformer_preds_on_batch(model, fasta_extractor, batch_df, track_idx):
  # prepare one-hot batch
  one_hot_seq_col = []
  for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df)):
    target_interval = kipoiseq.Interval(row["Chromosome"], row["Start"], row["End"]) 
    sequence_one_hot = one_hot_encode(fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH)))
    one_hot_seq_col.append(sequence_one_hot)
  
  print("Length of one-hot col: ", len(one_hot_seq_col))
  print(len(batch_df), SEQUENCE_LENGTH, 4)
  one_hot_seq_batch = np.reshape(one_hot_seq_col, (len(batch_df), SEQUENCE_LENGTH, 4))
  predictions = model.predict_on_batch(one_hot_seq_batch)['human'][:,:,track_idx]
  
  mean_preds = []
  for idx, row in batch_df.iterrows():
    output_start = (row["Start"] + row["End"])/2 - OUTPUT_SEQ_LENGTH/2
    output_end = (row["Start"] + row["End"])/2 + OUTPUT_SEQ_LENGTH/2
    interval_bins = np.linspace(output_start, output_end, num=896, endpoint=False)
    exp_output_start = np.abs(interval_bins - row["Start"]).argmin()
    exp_output_end = np.abs(interval_bins - row["End"]).argmin()
    ret_pred = predictions[idx, exp_output_start:exp_output_end]
    mean_preds.append([row["tss_idx"], ret_pred.mean()])
  
  return mean_preds

if __name__=="__main__":
  pyfaidx.Faidx(fasta_file)
  fasta_extractor = FastaStringExtractor(fasta_file)
  model = Enformer(model_path)
  enformer_track_idx = 4873 # CAGE:Astrocyte - cerebellum

  astrocyte_file = "/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/data/leaveout_celltypes/cerebellum_1.csv"
  astrocyte_df = pd.read_csv(astrocyte_file)

  tss_file = "/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/data/leaveout_celltypes/cerebellum_1.tss.npy"
  tss_array = np.load("/pmglocal/alb2281/repos/get_model/get_model/baselines/enformer/data/leaveout_celltypes/cerebellum_1.tss.npy")

  # get indices for tss True on either strand
  tss_indices = np.where(tss_array[:,0] | tss_array[:,1])[0]

  # filter astrocyte_df to rows where TSS is True on either strand
  astrocyte_df = astrocyte_df.iloc[tss_indices]
  astrocyte_df["tss_idx"] = astrocyte_df.index
  
  print(f"Predicting for {len(astrocyte_df)} regions.")
    
  batch_preds = []
  num_batches = 0

  # interrupted at start=3600
  for start in tqdm(range(3600, len(astrocyte_df), BATCH_SIZE)):
    end = start + BATCH_SIZE
    batch_df = astrocyte_df[start:end].reset_index()
    mean_pred = compute_enformer_preds_on_batch(model, fasta_extractor, batch_df, enformer_track_idx)
    batch_preds.append(mean_pred)

    if (num_batches + 1) % SAVE_EVERY_N_BATCHES == 0:
      flat_preds = [item for sublist in batch_preds for item in sublist]
      np.save(f"{output_dir}/astrocyte_cerebellum_1_preds_example_{start}.npy", flat_preds)
      batch_preds = []

    num_batches += 1

  if batch_preds:
    flat_preds = [item for sublist in batch_preds for item in sublist]
    np.save(f"{output_dir}/astrocyte_cerebellum_1_preds_example_{start}.npy", flat_preds)
