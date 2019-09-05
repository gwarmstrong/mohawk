# from mohawk.data_downloader import data_downloader
# from mohawk.simulation import simulate_from_genomes, id_to_lineage
import os
import torch
import pandas as pd
from mohawk.trainer import trainer
from mohawk.models import BaseModel, SmallConvNet, ConvNet2, ConvNetAvg
from mohawk.simulation import id_to_lineage

# trial_ids = [
#              'GCF_000011545.1',  # Burkholderia pseudomallei K96243
#              'GCF_001999985.1',  # Helicobacter bilis
#              'GCF_000953655.1',  # Legionella hackeliae
#              'GCF_000590475.1',  # Pseudomonas stutzeri
#              'GCF_002209165.2',  # Staphylococcus sciuri
#              'GCF_001558415.2',  # Vibrio fluvialis
#              ]
# all ids not in validation
trial_ids = None
validation_ids = None 
# try another thing... train on all references for these clades that are not
# in validation id's (make sure to exclude strain)

level = 'genus'

trial_directory = '/panfs/panfs1.ucsd.edu/panscratch/garmstro/mohawk/genome_annotation/ref107/raw_01'
taxonomy_file = '/panfs/panfs1.ucsd.edu/panscratch/garmstro/mohawk/genome_annotation/rank_tids.tsv'

trial_ids = [id_[:-4] for id_ in os.listdir(trial_directory)]
trial_classes = id_to_lineage(trial_ids, level, channel=taxonomy_file)
class_counts = pd.Series(trial_classes).value_counts().to_dict()
distribution_numerator = [1 / class_counts[class_] for class_ in trial_classes]
distribution_denominator = sum(distribution_numerator)
distribution = [numerator / distribution_denominator for numerator in
                distribution_numerator]

trial_directory = '/panfs/panfs1.ucsd.edu/panscratch/garmstro/mohawk/genome_annotation/ref107/raw_01'
taxonomy_file = '/panfs/panfs1.ucsd.edu/panscratch/garmstro/mohawk/genome_annotation/rank_tids.tsv'

# n_samples = len(trial_ids)
# distribution = [1/n_samples] * n_samples
total_reads = 200# 00 * 15  # * 10
length = 150
train_ratio = 0.8
seed = 1234
batch_size = 64
weight = False  # True
distribution_noise = False

external_validation_params = {
    'external_validation_ids': validation_ids,
    'n_external_validation_reads': 200,# 000,  # * 3,
    'external_validation_distribution': [1/6] * 6,  # TODO make safe to
    # changes in ids
}

train_kwargs = {'gpu': torch.cuda.is_available(),
                'summary_interval': 1,
                'epochs': 1000,
                'summarize': True,
                'learning_rate': 0.0001,
                'log_dir': '../runs'
                }
summary_kwargs = {'classify_threshold': 0.8,
                  'concise': True}

print("CUDA available: {}".format(train_kwargs['gpu']))

# file_paths = data_downloader(trial_ids, genomes_directory=trial_directory)
# #print(file_paths)
# reads, classes = simulate_from_genomes(trial_ids, [1/6]*6, 10000, 4,
#                                        sequence_directory=trial_directory)

# print(reads[:5], classes[:5])
# print(reads[-5:], classes[-5:])

model = trainer(ConvNetAvg, distribution, total_reads, length, train_ratio,
                id_list=None, level=level, batch_size=batch_size,
                data_directory=trial_directory, random_seed=seed,
                weight=weight, distribution_noise=distribution_noise,
                train_kwargs=train_kwargs, summary_kwargs=summary_kwargs, taxonomy_mapping=taxonomy_file,
                channel=taxonomy_file)
print(model)
