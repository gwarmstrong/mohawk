# from mohawk.data_downloader import data_downloader
# from mohawk.simulation import simulate_from_genomes, id_to_lineage
import torch
from mohawk.trainer import trainer
from mohawk.models import BaseModel, SmallConvNet, ConvNet2, ConvNetAvg

# trial_ids = [
#              'GCF_000011545.1',  # Burkholderia pseudomallei K96243
#              'GCF_001999985.1',  # Helicobacter bilis
#              'GCF_000953655.1',  # Legionella hackeliae
#              'GCF_000590475.1',  # Pseudomonas stutzeri
#              'GCF_002209165.2',  # Staphylococcus sciuri
#              'GCF_001558415.2',  # Vibrio fluvialis
#              ]
# all ids not in validation
trial_ids = ['GCF_000006745.1', 'GCF_000006765.1', 'GCF_000007565.2',
             'GCF_000007645.1', 'GCF_000007805.1', 'GCF_000008485.1',
             'GCF_000008525.1', 'GCF_000009865.1', 'GCF_000010125.1',
             'GCF_000011545.1', 'GCF_000011705.1', 'GCF_000012245.1',
             'GCF_000016565.1', 'GCF_000017705.1', 'GCF_000025085.1',
             'GCF_000091465.1', 'GCF_000091785.1', 'GCF_000091985.1',
             'GCF_000185885.1', 'GCF_000196095.1', 'GCF_000200595.1',
             'GCF_000213805.1', 'GCF_000217675.1', 'GCF_000219605.1',
             'GCF_000259255.1', 'GCF_000259275.1', 'GCF_000332735.1',
             'GCF_000349975.1', 'GCF_000354175.2', 'GCF_000397205.1',
             'GCF_000412695.1', 'GCF_000590475.1', 'GCF_000706685.1',
             'GCF_000761155.1', 'GCF_000763535.1', 'GCF_000772105.1',
             'GCF_000801275.2', 'GCF_000816085.1', 'GCF_000818015.1',
             'GCF_000953135.1', 'GCF_000953655.1', 'GCF_000959245.1',
             'GCF_001028645.1', 'GCF_001411495.1', 'GCF_001432245.1',
             'GCF_001460635.1', 'GCF_001534745.1', 'GCF_001558415.2',
             'GCF_001559115.2', 'GCF_001597285.1', 'GCF_001602095.1',
             'GCF_001618885.1', 'GCF_001654435.1', 'GCF_001677275.1',
             'GCF_001685465.1', 'GCF_001902315.1', 'GCF_001913135.1',
             'GCF_001999985.1', 'GCF_002101335.1', 'GCF_002196515.1',
             'GCF_002208805.2', 'GCF_002209165.2', 'GCF_002215135.1',
             'GCF_002240035.1']
validation_ids = [
                  'GCF_000959725.1',  # Burkholderia gladioli
                  'GCF_000007905.1',  # Helicobacter hepaticus
                  'GCF_000512355.1',  # Legionella oakridgensis
                  'GCF_000237065.1',  # Pseudomonas fluorescens
                  'GCF_000013425.1',  # Staphylococcus aureus
                  'GCF_001547935.1',  # Vibrio tritonius
                  ]

# try another thing... train on all references for these clades that are not
# in validation id's (make sure to exclude strain)

trial_directory = '../data/trial_download'
n_samples = len(trial_ids)
distribution = [1/n_samples] * n_samples
total_reads = 200#00 * 15  # * 10
length = 150
train_ratio = 0.8
seed = 1234
batch_size = 64
weight = True

external_validation_params = {
    'external_validation_ids': validation_ids,
    'n_external_validation_reads': 20, # 0000,  # * 3,
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

model = trainer(ConvNetAvg, trial_ids, distribution, total_reads, length,
                train_ratio, data_directory=trial_directory, random_seed=seed,
                batch_size=batch_size,
                weight=weight,
                **external_validation_params,
                train_kwargs=train_kwargs,
                summary_kwargs=summary_kwargs)
print(model)
