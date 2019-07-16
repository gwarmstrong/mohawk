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
        expected_ids = ['seq1__idx_0',
                        'seq1__idx_1',
                        'seq1__idx_2',
                        'seq1__idx_3',
                        'seq1__idx_4',
                        'seq2__idx_0',
                        'seq3__idx_0']
        list_seqs = [[l1 for l1 in l2] for l2 in seqs]
        self.assertListEqual(list_seqs, expected_seqs)
        self.assertListEqual(ids, expected_ids)
