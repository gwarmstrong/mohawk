import numpy as np
from mohawk.data_downloader import data_downloader
from mohawk.simulation import simulate_from_genomes, id_to_lineage
from mohawk.dataset import SequenceDataset, one_hot_encode_classes
from typing import Optional, List
import torch.nn
from torch.utils.data import DataLoader


def trainer(model: torch.nn.Module,
            id_list: List[str],
            distribution: List[float],
            total_reads: int,
            length: int,
            train_ratio: float,
            level: Optional[str] = 'genus',
            channel: Optional[str] = 'representative',
            directory: Optional[str] = None,
            random_seed: Optional[int] = None,
            model_kwargs: Optional[dict] = None,
            train_kwargs: Optional[dict] = None):

    data_downloader(id_list,
                    genomes_directory=directory,
                    channel=channel)

    reads, ids = simulate_from_genomes(id_list, distribution, total_reads,
                                       length, channel, directory, random_seed)

    classes = id_to_lineage(ids, level, channel)

    classes_ohe, encoder = one_hot_encode_classes(classes)

    # split into train and validation
    # TODO clade exclusion
    num_samples = len(classes)
    train_indices, val_indices = train_val_split(num_samples, train_ratio,
                                                 random_seed=random_seed)

    # create Dataset and DataLoader for train and validation
    train_components = prep_reads_classes_ids(reads, classes_ohe, ids,
                                              train_indices)
    val_components = prep_reads_classes_ids(reads, classes_ohe, ids,
                                            val_indices)

    train_dataset = SequenceDataset(*train_components)
    val_dataset = SequenceDataset(*val_components)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    # train model
    trained_model = train(model, train_dataloader, val_dataloader,
                          model_kwargs, **train_kwargs)

    # write out final model and some summary information
    # TODO have a seperate directory for finished model
    # TODO as well as prefix/suffix option for naming
    trained_model.summarize(directory)


def train(model: torch.nn.Module,
          train_data: DataLoader,
          val_data: Optional[DataLoader],
          model_kwargs: Optional[dict],
          **kwargs):

    mod = model(**model_kwargs)

    # TODO write training function

    return mod


def prep_reads_classes_ids(reads, classes, ids, indices):
    sub_reads = reads[indices]
    classes_array = np.array(classes)
    sub_classes = classes_array[indices]
    ids_array = np.array(ids)
    sub_ids = ids_array[indices]

    return sub_reads, sub_classes, sub_ids


def train_val_split(num_samples, train_ratio, random_seed=None):
    indices = np.arange(num_samples)
    if random_seed is not None:
        np.random.seed(random_seed + 1)
    permutation = np.random.permutation(indices)
    train_max_index = int(num_samples * train_ratio)
    train_indices = permutation[:train_max_index]
    test_indices = permutation[train_max_index:]
    return train_indices, test_indices