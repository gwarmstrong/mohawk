import unittest
from mohawk.classify import ensure_lengths


class TestClassifyHelpers(unittest.TestCase):

    def test_ensure_lengths(self):
        length = 3
        seq1 = [0, 1, 2, 1, 0, 3, 1]
        seq2 = [0, 2, 1]
        seq3 = [0, 2]
        ids = ['seq1', 'seq2', 'seq3']
        input_seqs = [seq1, seq2, seq3]
        seqs, ids = ensure_lengths(list(zip(input_seqs, ids)), length=length)

        expected_seqs = [[0, 1, 2],
                         [1, 2, 1],
                         [2, 1, 0],
                         [1, 0, 3],
                         [0, 3, 1],
                         [0, 2, 1],
                         [0, 2, 0]]
        expected_ids = ['seq1'] * 5 + ['seq2'] + ['seq3']
        self.assertListEqual(seqs, expected_seqs)
        self.assertListEqual(ids, expected_ids)
