from torch.utils.data import Dataset


class SequenceDataset(Dataset):

    def __init__(self, reads, classes, encoder=None):
        super(self, SequenceDataset).__init__()
        if len(reads) != len(classes):
            raise ValueError("There must be one class for every read.")

        self.reads = reads
        self.classes = classes
        self.encoder = encoder

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        return {'read': self.reads[idx], 'label': self.classes[idx]}

