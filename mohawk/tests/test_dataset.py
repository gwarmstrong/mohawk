import unittest
import numpy as np

from mohawk.dataset import SequenceDataset


class TestSequenceDataset(unittest.TestCase):

    def setUp(self) -> None:
        reads = np.array([[0, 1, 2, 3], [0, 2, 3, 1], [3, 2, 1, 0],
                          [1, 0, 3, 2]])
        classes = ['a', 'b', 'c', 'a']
        ids = ['A', 'B', 'C', 'D']
        self.dataset = SequenceDataset(reads, classes, ids)

    def test_read_encoding(self):
        expected_reads = [[[1, 0, 0, 0],
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

        expected_reads = np.array([np.array(array).T for array in
                                   expected_reads])
        observed_reads = np.array(self.dataset.reads)

        self.assertTrue((expected_reads == observed_reads).all())

    def test_label_encoding(self):
        expected_labels = [0, 1, 2, 0]
        observed_labels = list(self.dataset.labels)
        self.assertListEqual(expected_labels, observed_labels)

    def test_length(self):
        self.assertEquals(len(self.dataset), 4)

    def test_get_item(self):
        # TODO do for all items?
        last_item = self.dataset[3]
        self.assertEquals(0, last_item['label'])
        self.assertEquals('a', last_item['label_english'])
        last_read = last_item['read']
        expected_read = np.array([[0, 1, 0, 0],
                                  [1, 0, 0, 0],
                                  [0, 0, 0, 1],
                                  [0, 0, 1, 0]]).T
        self.assertTrue((last_read == expected_read).all())
