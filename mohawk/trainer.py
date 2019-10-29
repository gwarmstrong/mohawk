import os
import time
import numpy as np
import pandas as pd
from mohawk.data_downloader import data_downloader
from mohawk.simulation import simulate_from_genomes
from mohawk.dataset import SequenceDataset
from typing import Optional, List
from mohawk.models import BaseModel, SmallConvNet, ConvNet2, ConvNetAvg, \
    ConvNetAvg2, ConvNetAvg3, ConvNetAvg4, ConvNetAvg5, ConvNetAvg6
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# TODO current file_list, taxonomy setup needs to be re-worked...
def trainer(model: BaseModel, total_reads: int, length: int,
            train_ratio: float, id_list: Optional[List[str]],
            distribution: List[float], class_list: Optional[List] = None,
            metadata: Optional[str] = None, batch_size: Optional[int] = 1,
            data_directory: Optional[str] = None,
            random_seed: Optional[int] = None,
            external_validation_ids: Optional[List[str]] = None,
            n_external_validation_reads: Optional[int] = None,
            external_validation_distribution: Optional[List[float]] = None,
            external_validation_classes: Optional[List] = None,
            start_time: Optional[float] = None,
            append_time: Optional[bool] = True,
            additional_hparams: Optional[dict] = None,
            model_kwargs: Optional[dict] = None,
            train_kwargs: Optional[dict] = None,
            summary_kwargs: Optional[dict] = None) -> BaseModel:
    """

    Parameters
    ----------
    model
        An model class to use for training
    total_reads
        The total number of reads to simulate for training
    length
        The length of reads to simulate for training
    train_ratio
        The portion of simulated reads that should be used for training
    id_list
        The list of genome ids to use for training
    distribution
        The relative amount of each id in id_list to use
    class_list
        The class of each id in id_list
    metadata
        Path to metadata containing a '# assembly_accession' and 'ftp_path'
        column (NCBI assembly summary format)
    batch_size
        Size of batches for neural network training
    data_directory
        Directory that genome data is saved in, or should be saved in for
        downloaded data
    random_seed
        Seed for the random number generator
    external_validation_ids
        Genome ids to use for validating that should be exclusive with
        training id_list
    n_external_validation_reads
        how many reads to sample from the external validation ids
    external_validation_distribution
        How to distribute the reads amongst the external validation ID's
    external_validation_classes
        The class of each id in external_validation_ids
    start_time
        An object for tracking amount of time since training has started.
        Can be passed in for convenience, but will be initialized if not
        provided
    append_time
        Indicates whether the time should be appended to the log_dir,
        cannot be False if log_dir is None
    additional_hparams
        hyper-parameters that can be passed into to be saved with the model
    model_kwargs
        kwargs to be passed to the `model`
    train_kwargs
        kwargs to be passed to the training function of the `model`
    summary_kwargs
        kwargs to be passed to the summary funciton of the `model`



    Returns
    -------

    trained_model
        The model that has been trained by your trainer

    """

    if additional_hparams is None:
        additional_hparams = dict()
    if model_kwargs is None:
        model_kwargs = dict()
    if train_kwargs is None:
        train_kwargs = dict()
    if summary_kwargs is None:
        summary_kwargs = dict()
    if start_time is None:
        start_time = time.time()

    log_dir = train_kwargs.pop('log_dir', None)
    if append_time and not log_dir:
        # this error is to avoid multiple models accidentally getting the
        #  written to the same directory. Take care to use different
        #  log_dirs if using append_time=False
        raise ValueError("'append_time' can cannot be False when 'log_dir' "
                         "is None")
    log_dir = model.get_log_dir(log_dir, append_time=append_time)

    # If id_list is not None, use the specified id's
    if id_list is not None:
        file_list = data_downloader(id_list, output_directory=data_directory,
                                    metadata=metadata)
    # if id_list _is_ None, just use whatever is in the directory
    elif os.path.exists(data_directory) and \
            len(os.listdir(data_directory)) > 0:
        file_list = [os.path.join(data_directory, file_) for file_ in
                     os.listdir(data_directory)]
    else:
        raise FileExistsError('Data directory must exist and contain '
                              'data if `id_list` is not supplied.')

    reads, ids = simulate_from_genomes(distribution, total_reads, length,
                                       file_list, data_directory, random_seed)

    id_depths = [round(val * total_reads) for val in distribution]
    if class_list is None:
        class_list = id_list
    list_of_classes = [[class_] * depth for class_, depth in
                       zip(class_list, id_depths)]
    class_list = [item for sublist in list_of_classes for item in sublist]

    external_validation = False
    external_classes = None
    if external_validation_ids is not None and \
            n_external_validation_reads is not None and \
            external_validation_distribution is not None and \
            external_validation_classes is not None:
        data_downloader(external_validation_ids,
                        output_directory=data_directory, metadata=metadata)
        external_reads, external_ids = simulate_from_genomes(
            external_validation_distribution, n_external_validation_reads,
            length, external_validation_ids, data_directory, random_seed + 5)

        external_validation = True

        ext_depths = [round(val * n_external_validation_reads) for val in
                      external_validation_distribution]
        list_of_ext_classes = [[class_] * depth for class_, depth in zip(
            external_validation_classes, ext_depths)]

        external_classes = [item for sublist in list_of_ext_classes for item
                            in sublist]

    elif external_validation_ids is not None or \
            n_external_validation_reads is not None or \
            external_validation_distribution is not None or \
            external_validation_classes is not None:
        raise ValueError('If any external validation parameters are '
                         'specified, all must be specified.')

    # split into train and validation
    train_indices, val_indices = train_val_split(total_reads, train_ratio,
                                                 random_seed=random_seed)

    # create Dataset and DataLoader for train and validation
    train_dataloader = prepare_dataloader(reads,
                                          class_list,
                                          ids,
                                          batch_size=batch_size,
                                          indices=train_indices)
    val_dataloader = prepare_dataloader(reads,
                                        class_list,
                                        ids,
                                        batch_size=batch_size,
                                        indices=val_indices)
    if external_validation:
        external_dataloader = prepare_dataloader(external_reads,
                                                 external_classes,
                                                 external_ids,
                                                 batch_size=batch_size)
    else:
        external_dataloader = None

    training_model = model(n_classes=len(set(class_list)),
                           seed=random_seed,
                           **model_kwargs)

    # conv1d complains if there are float's instead of doubles <- could slow
    # down training
    training_model.double()

    writer = SummaryWriter(log_dir=log_dir)

    if train_kwargs['gpu']:
        training_model.cuda()

    # TODO prefix/suffix option for naming
    training_model.fit(train_dataloader,
                       log_dir=log_dir,
                       val_dataset=val_dataloader,
                       external_dataset=external_dataloader,
                       seed=random_seed,
                       summary_kwargs=summary_kwargs,
                       log_dir_append_time=False,
                       writer=writer,
                       start_time=start_time,
                       **train_kwargs)

    end_time = time.time()
    mod = training_model

    hparam_dict = dict()
    hparam_train_kwargs = ['learning_rate', 'epochs', 'gpu']
    hparam_dict.update({kwarg: train_kwargs[kwarg] for
                        kwarg in hparam_train_kwargs})
    hparam_dict.update(model_kwargs)
    hparam_dict.update({'model_type': model.__name__,
                        'random-seed': mod.seed})
    hparam_dict.update(additional_hparams)

    metric_dict = {'best-val-accuracy': mod.best_val_accuracy,
                   'best-val-loss': mod.best_val_loss,
                   'best-val-epoch': mod.best_val_epoch,
                   'best-val-train-accuracy': mod.best_val_train_accuracy,
                   'best-val-train-loss': mod.best_val_train_loss,
                   'best-val-time': mod.best_val_time,
                   'train-dataset-length': len(train_dataloader.dataset),
                   'val-dataset-length': len(val_dataloader.dataset),
                   'total-time': end_time - start_time,
                   }
    writer.add_hparams(hparam_dict, metric_dict)

    writer.close()

    return training_model


def prepare_weights(dataloader):
    all_labels = dataloader.dataset.labels
    all_labels = pd.Series(all_labels).value_counts().sort_index()
    # will error if no labels present in training data for given class
    inv = (1 / all_labels)
    weights = inv / inv.sum()
    return weights


def prepare_dataloader(reads, classes=None, ids=None, indices=None,
                       batch_size=1,
                       shuffle=False):
    # TODO right now, cannot use NONE for classes and ids if indices is not
    #  None -> should fix this so subsetting can be done of classificaiton data
    if indices is not None:
        components = prep_reads_classes_ids(reads, classes, ids,
                                            indices)
    else:
        components = reads, classes, ids

    dataset = SequenceDataset(*components)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def prep_reads_classes_ids(reads, classes, ids, indices):
    # TODO could error if indices is full/empty
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


# TODO incorporate defaults from mohawk train
def train_helper(model_name, genome_ids,
                 external_validation_ids=None,
                 metadata=None,
                 lr=0.0001,
                 epochs=100,
                 summarize=True,
                 log_dir='runs/',
                 summary_interval=5,
                 train_ratio=0.8,
                 length=150,
                 seed=0,
                 concise_summary=True,
                 gpu=False,
                 batch_size=64,
                 data_dir=os.curdir,
                 additional_hyper_parameters=None,
                 append_time=True):
    start_time = time.time()
    model = model_names_to_obj[model_name]
    id_list, distribution, classes, n_reads = id_file_loader(genome_ids)
    if external_validation_ids is not None:
        ext_ids, ext_dist, ext_classes, ext_reads = id_file_loader(
            external_validation_ids)
    else:
        ext_ids, ext_dist, ext_classes, ext_reads = None, None, None, None
    train_kwargs = {'gpu': gpu,
                    'learning_rate': lr,
                    'summary_interval': summary_interval,
                    'epochs': epochs,
                    'summarize': summarize,
                    'log_dir': log_dir,
                    }
    summary_kwargs = {'concise': concise_summary}
    if additional_hyper_parameters is not None:
        additional_hparams = parse_hparams_file(additional_hyper_parameters)
    else:
        additional_hparams = None
    trainer(model, n_reads, length, train_ratio, id_list=id_list,
            metadata=metadata, distribution=distribution, class_list=classes,
            batch_size=batch_size, data_directory=data_dir, random_seed=seed,
            external_validation_ids=ext_ids,
            n_external_validation_reads=ext_reads,
            external_validation_distribution=ext_dist,
            external_validation_classes=ext_classes,
            start_time=start_time, train_kwargs=train_kwargs,
            summary_kwargs=summary_kwargs,
            additional_hparams=additional_hparams,
            append_time=append_time)

    return model


def parse_hparams_file(fp):
    df = pd.read_csv(fp, sep='\t', index_col=0)
    dict_ = df.to_dict()['0']
    out_dict = dict()
    for key, val in dict_.items():
        # try to make the key numerical, but if not able to, leave as str
        try:
            out_dict[key] = float(val)
        except KeyError:
            out_dict[key] = val
    return out_dict


model_names_to_obj = {'SmallConvNet': SmallConvNet,
                      'ConvNet2': ConvNet2,
                      'ConvNetAvg': ConvNetAvg,
                      'ConvNetAvg2': ConvNetAvg2,
                      'ConvNetAvg3': ConvNetAvg3,
                      'ConvNetAvg4': ConvNetAvg4,
                      'ConvNetAvg5': ConvNetAvg5,
                      'ConvNetAvg6': ConvNetAvg6
                      }


def id_file_loader(genome_ids):
    id_df = pd.read_csv(genome_ids, sep='\t')
    id_list = id_df['id']
    n_reads_by_genome = id_df['n_reads']
    n_reads = int(n_reads_by_genome.sum())
    distribution = n_reads_by_genome / n_reads
    if 'class' in id_df.columns:
        classes = id_df['class']
    else:
        classes = None

    return id_list, distribution, classes, n_reads
