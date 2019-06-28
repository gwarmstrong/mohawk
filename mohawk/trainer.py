import numpy as np
from mohawk.data_downloader import data_downloader
from mohawk.simulation import simulate_from_genomes, id_to_lineage
from mohawk.dataset import SequenceDataset, encode_classes # TODO should go?
from typing import Optional, List
from torch import nn
from torch.utils.data import DataLoader


def trainer(model: nn.Module,
            id_list: List[str],
            distribution: List[float],
            total_reads: int,
            length: int,
            train_ratio: float,
            level: Optional[str] = 'genus',
            channel: Optional[str] = 'representative',
            data_directory: Optional[str] = None,
            summary_directory: Optional[str] = None,
            random_seed: Optional[int] = None,
            external_validation_ids: Optional[List[str]] = None,
            n_external_validation_reads: Optional[int] = None,
            external_validation_distribution: Optional[List[float]] = None,
            model_kwargs: Optional[dict] = dict(),
            train_kwargs: Optional[dict] = dict(),
            summary_kwargs: Optional[dict] = dict()):

    data_downloader(id_list,
                    genomes_directory=data_directory,
                    channel=channel)

    reads, ids = simulate_from_genomes(id_list, distribution, total_reads,
                                       length, channel, data_directory,
                                       random_seed)

    if external_validation_ids is not None and \
            n_external_validation_reads is not None and \
            external_validation_distribution is not None:
        data_downloader(external_validation_ids,
                        genomes_directory=data_directory,
                        channel=channel)
        external_reads, external_ids = simulate_from_genomes(
            external_validation_ids,
            external_validation_distribution,
            n_external_validation_reads,
            length,
            channel,  # presumably should be available in the same channel
            data_directory,
            random_seed + 5
            )
    elif external_validation_ids is not None or \
            n_external_validation_reads is not None or \
            external_validation_distribution is not None:
        raise ValueError('If any external validation parameters are '
                         'specified, all must be specified.')
    else:
        external_reads = None
        external_ids = None

    classes = id_to_lineage(ids, level, channel)

    if external_ids is not None:
        external_classes = id_to_lineage(external_ids, level, channel)
    else:
        external_classes = None

    # TODO is classes ohe actually necessary?
    # classes_enc, encoder = encode_classes(classes)

    # split into train and validation
    # TODO clade exclusion validation
    num_samples = len(classes)
    train_indices, val_indices = train_val_split(num_samples, train_ratio,
                                                 random_seed=random_seed)

    # create Dataset and DataLoader for train and validation
    train_components = prep_reads_classes_ids(reads, classes, ids,
                                              train_indices)
    val_components = prep_reads_classes_ids(reads, classes, ids,
                                            val_indices)

    if external_classes is not None:
        external_components = external_reads, external_classes, external_ids
                              # prep_reads_classes_ids(external_reads,
                              #                        external_classes,
                              #                        external_ids)
    else:
        external_components = None

    train_dataset = SequenceDataset(*train_components)
    val_dataset = SequenceDataset(*val_components)

    if external_components is not None:
        external_dataset = SequenceDataset(*external_components)
    else:
        external_dataset = None

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    if external_dataset is not None:
        external_dataloader = DataLoader(external_dataset,
                                         batch_size=64,
                                         shuffle=True)
    else:
        external_dataloader = None

    training_model = model(length=length,
                           n_classes=len(set(classes)),
                           seed=random_seed,
                           **model_kwargs)

    # conv1d complains if there are float's instead of doubles <- could slow
    # down training
    training_model.double()
    # TODO call model.cuda() here maybe?
    if train_kwargs['gpu']:
        training_model.cuda()

    training_model.fit(train_dataloader,
                       val_dataset=val_dataloader,
                       external_dataset=external_dataloader,
                       seed=random_seed,
                       log_dir=summary_directory,
                       summary_kwargs=summary_kwargs,
                       **train_kwargs)

    # write out final model and some summary information
    # TODO have a seperate directory for finished model
    # TODO as well as prefix/suffix option for naming
    # training_model.summarize(summary_directory,
    #                          train_dataset=train_dataset,
    #                          val_dataset=val_dataset,
    #                          **summary_kwargs)

    return training_model


def prep_reads_classes_ids(reads, classes, ids, indices):
    # if indices are not specified, keep everything
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