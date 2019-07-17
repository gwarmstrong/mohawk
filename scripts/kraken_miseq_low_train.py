import torch
import pandas as pd
from mohawk.trainer import trainer
from mohawk.models import ConvNetAvg
from mohawk.simulation import id_to_lineage


# from KRAKEN ACCURACY.README distributed with low:
# - Bacillus cereus VD118
# - Citrobacter freundii 47N
# - Enterobacter cloacae
# - Klebsiella pneumoniae NES14
# - Mycobacterium abscessus 6G-0125-R
# - Proteus vulgaris 66N
# - Rhodobacter sphaeroides 2.4.1
# - Staphylococcus aureus ST22
# - Salmonella enterica Montevideo str. N19965
# - Vibrio cholerae CP1032(5)

# currently using only representative genomes
train_ids = [
    # Bacillus cereus ids:
    'GCF_000007825.1',
    # Citrobacter freundii ids:
    'GCF_000648515.1',
    # Enterobacter cloacae
    'GCF_000025565.1',
    # Klebsiella pneumoniae
    'GCF_000240185.1',
    # Mycobacteroides abscessus
    'GCF_000069185.1',
    # Proteus vulgaris
    'GCF_002591155.1', 'GCF_003812525.1',
    # Rhodobacter sphaeroides
    'GCF_000016405.1',
    # Staphylococcus aureus
    'GCF_000013425.1',
    # Salmonella enterica
    'GCF_000006945.2', 'GCF_000195995.1',
    # Vibrio cholerae
    'GCF_000006745.1'
]

level = 'genus'
channel = 'complete'

trial_classes = id_to_lineage(train_ids, level, channel=channel)
class_counts = pd.Series(trial_classes).value_counts().to_dict()
distribution_numerator = [1 / class_counts[class_] for class_ in trial_classes]
distribution_denominator = sum(distribution_numerator)
distribution = [numerator / distribution_denominator for numerator in
                distribution_numerator]

trial_directory = 'data/kraken_train_download'
n_samples = len(train_ids)
total_reads = 200 # 00 * 15  # * 10
length = 150
train_ratio = 0.8
seed = 1234
batch_size = 64
weight = False  # True
distribution_noise = False

train_kwargs = {'gpu': torch.cuda.is_available(),
                'summary_interval': 1,
                'epochs': 500,
                'summarize': True,
                'learning_rate': 0.0001,
                'log_dir': 'runs',
                'save_interval': 10
                }

summary_kwargs = {'concise': True}

print("CUDA available: {}".format(train_kwargs['gpu']))

model = trainer(ConvNetAvg, train_ids, distribution, total_reads, length,
                train_ratio,
                channel=channel,
                data_directory=trial_directory,
                random_seed=seed,
                level=level,
                batch_size=batch_size,
                weight=weight,
                distribution_noise=distribution_noise,
                train_kwargs=train_kwargs,
                summary_kwargs=summary_kwargs)
print(model)
