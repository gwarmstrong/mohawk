# from mohawk.data_downloader import data_downloader
# from mohawk.simulation import simulate_from_genomes, id_to_lineage
from mohawk.trainer import trainer
from mohawk.models import BaseModel, SmallConvNet

trial_ids = ['GCF_900128725.1', 'GCF_000590475.1', 'GCF_000953655.1',
             'GCF_001558415.2', 'GCF_002209165.2', 'GCF_001999985.1']
trial_directory = '../data/trial_download'
distribution = [1/6]*6
total_reads = 100000
length = 150
train_ratio = 0.8

# file_paths = data_downloader(trial_ids, genomes_directory=trial_directory)
# #print(file_paths)
# reads, classes = simulate_from_genomes(trial_ids, [1/6]*6, 10000, 4,
#                                        sequence_directory=trial_directory)

# print(reads[:5], classes[:5])
# print(reads[-5:], classes[-5:])

model = trainer(SmallConvNet, trial_ids, distribution, total_reads, length,
                train_ratio, data_directory=trial_directory, random_seed=1234)
print(model)
