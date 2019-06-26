from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class SequenceDataset(Dataset):

    def __init__(self, reads, classes, encoder=None):
        super(self, SequenceDataset).__init__()
        if len(reads) != len(classes):
            raise ValueError("There must be one class for every read.")

        read_length = len(reads[0])
        n_bases = 4

        # will encode each sequence of length l as a sequence of length 4 * l
        # will mean we should take
        sequence_ohe = OneHotEncoder(categories=[np.arange(n_bases) for _ in
                                                 range(read_length)])

        reads_transformed = sequence_ohe.fit_transform(reads)

        reads_matrices = [seq.reshape(read_length, n_bases) for seq in
                          reads_transformed]

        self.reads = reads_matrices
        self.classes = classes
        self.encoder = encoder

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        return {'read': self.reads[idx], 'label': self.classes[idx]}


def one_hot_encode_classes(classes):
    encoder = OneHotEncoder()

    classes = encoder.fit_transform(classes)

    return classes, encoder
