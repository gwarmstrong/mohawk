from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np


class SequenceDataset(Dataset):
    """
    Dataset that prepares a set of quaternary sequences into matrices of
    one-hot encoding matrices,

    e.g.,

    [[0, 1, 2, 3], [0, 2, 3, 1], [3, 2, 1, 0], [1, 0, 3, 2]]

    ->

    [[[1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1]],
     [[1, 0, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
      [0, 1, 0, 0]],
     [[0, 0, 0, 1],
      [0, 0, 1, 0],
      [0, 1, 0, 0],
      [1, 0, 0, 0]],
     [[0, 1, 0, 0],
      [1, 0, 0, 0],
      [0, 0, 0, 1],
      [0, 0, 1, 0]]]

    with and converts labels for each sequence into numerical labels

    e.g.,

    ['a', 'b', 'c', 'a'] -> [0, 1, 2, 0]


    Parameters
    ----------

    sequences:
        `n_sequences` x `sequence_length` matrix containing quaternary
        sequences
    classes:
        `n_sequences` labels for each sequence to be used for training
    ids:
        `n_sequences` labels of information for each sequence, could be
            more specific than `classes`

    Attributes
    ----------

    reads
        each entry contains a matrix representing the one-hot-encoded sequence
    labels
        a class label for each sequence (encoded as an integer)
    classes
        a class label for each sequence (encoded as input)
    label_encoder
        the encoder used to transform `classes` to `labels`
    sequence_encoder
        the encoder used to transform the input sequences into `reads`
    ids
        the id information supplied by `ids`

    """

    def __init__(self, sequences, classes, ids):
        if len(sequences) != len(classes):
            raise ValueError("There must be one class for every read.")

        sequence_length = len(sequences[0])

        # TODO could generalize
        n_bases = 4

        # will encode each sequence of length l as a sequence of length 4 * l
        # will mean we should take
        sequence_ohe = OneHotEncoder(categories=[np.arange(n_bases) for _ in
                                                 range(sequence_length)])

        reads_transformed = sequence_ohe.fit_transform(sequences.astype(int))

        reads_matrices = [np.array(seq.todense())
                            .reshape(sequence_length, n_bases)
                            .T
                          for seq in reads_transformed]

        label_enc = LabelEncoder()

        labels = label_enc.fit_transform(classes)

        self.reads = reads_matrices
        self.labels = labels
        self.classes = classes
        self.label_encoder = label_enc
        self.sequence_encoder = sequence_ohe
        self.ids = ids

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        # TODO could save on some memory by not loading `label_english`
        return {'read': self.reads[idx], 'label': self.labels[idx],
                'label_english': self.classes[idx]}
