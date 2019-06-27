from mohawk.data_downloader import data_downloader
from mohawk.simulation import simulate_from_genomes, id_to_lineage
from mohawk.dataset import SequenceDataset, one_hot_encode_classes


def trainer(id_list: List[str],
            distribution: List[float],
            total_reads: int,
            length: int,
            validation_ratio: float,
            level: Optional[str] = 'genus',
            channel: Optional[str] = 'representative',
            directory: Optional[str] = None,
            random_seed: Optional[int] = None):

    data_downloader(id_list,
                    genomes_directory=directory,
                    channel=channel)

    reads, ids = simulate_from_genomes(id_list, distribution, total_reads,
                                       length, channel, directory, random_seed)

    classes = id_to_lineage(ids, level, channel)

    classes_ohe, encoder = one_hot_encode_classes(classes)

    # shuffle data and split into train and validation (should eventually do
    # clade exclusion)

    # create Dataset and DataLoader

    # instantiate model

    # train model

    # write out final model and some summary information


