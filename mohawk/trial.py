# from mohawk.data_downloader import data_downloader
# from mohawk.simulation import simulate_from_genomes, id_to_lineage
import torch
from mohawk.trainer import trainer
from mohawk.models import BaseModel, SmallConvNet

trial_ids = [
             'GCF_000011545.1',  # Burkholderia pseudomallei K96243
             'GCF_001999985.1',  # Helicobacter bilis
             'GCF_000953655.1',  # Legionella hackeliae
             'GCF_000590475.1',  # Pseudomonas stutzeri
             'GCF_002209165.2',  # Staphylococcus sciuri
             'GCF_001558415.2',  # Vibrio fluvialis
             ]
validation_ids = [
                  'GCF_000959725.1',  # Burkholderia gladioli
                  'GCF_000007905.1',  # Helicobacter hepaticus
                  'GCF_000512355.1',  # Legionella oakridgensis
                  'GCF_000237065.1',  # Pseudomonas fluorescens
                  'GCF_000013425.1',  # Staphylococcus aureus
                  'GCF_001547935.1',  # Vibrio tritonius
                  ]
trial_directory = '../data/trial_download'
distribution = [1/6]*6
total_reads = 200000
length = 150
train_ratio = 0.8
seed = 1234

external_validation_params = {
    'external_validation_ids': validation_ids,
    'n_external_validation_reads': 50000,
    'external_validation_distribution': distribution,
}

train_kwargs = {'gpu': torch.cuda.is_available(),
                'summary_interval': 1,
                'epochs': 500
               }
summary_kwargs = {'classify_threshold': 0.25}
print("CUDA available: {}".format(train_kwargs['gpu']))

# file_paths = data_downloader(trial_ids, genomes_directory=trial_directory)
# #print(file_paths)
# reads, classes = simulate_from_genomes(trial_ids, [1/6]*6, 10000, 4,
#                                        sequence_directory=trial_directory)

# print(reads[:5], classes[:5])
# print(reads[-5:], classes[-5:])

model = trainer(SmallConvNet, trial_ids, distribution, total_reads, length,
                train_ratio, data_directory=trial_directory, random_seed=seed,
                **external_validation_params,
                train_kwargs=train_kwargs,
                summary_kwargs=summary_kwargs)
print(model)
