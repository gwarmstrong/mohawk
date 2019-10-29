__author__ = "Daniel McDonald"

import unittest
import numpy as np
import numpy.testing as npt
import skbio

from mohawk._format import (locus_generator,
                            extract_subsequences,
                            encode_sequence,
                            join_contigs,
                            sample_from_contig_set)


class FormattingTests(unittest.TestCase):
    def test_contig_locus_generator(self):
        length = 10
        start = 0
        stop = 20
        n_indices = 10

        def mock_function(start, stop, n):
            # ...ignoring n for mocking
            return np.arange(start, stop)

        exp = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        obs = locus_generator(start, stop, length, n_indices,
                              mock_function)
        npt.assert_equal(obs, exp)

    def test_extract_subsequences(self):
        #                    0  1  2  3  4  5  6  7  8  9 10 11
        sequence = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,
                             0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        positions = np.array([0, 4, 10])
        length = 5
        exp = np.array([[0, 0, 0, 1, 1],
                        [1, 1, 2, 2, 2],
                        [3, 3, 0, 0, 0]])
        obs = extract_subsequences(sequence, positions, length)
        npt.assert_equal(obs, exp)

        # verify we don't attempt to read "off the end"
        length = 10
        positions = np.array([0, 5, 18])
        with self.assertRaises(ValueError):
            extract_subsequences(sequence, positions, length)

    def test_encode_sequence(self):
        sequence = skbio.DNA('AAATTTGGGCCC')
        exp = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        obs = encode_sequence(sequence)
        npt.assert_equal(obs, exp)

        # note that skbio's DNA constructor protects us from lowercase

    def test_join_contigs(self):
        sequences = [skbio.DNA('AATTGG'), skbio.DNA('CCTTAA'),
                     skbio.DNA('ATAT')]

        #                    0123456789012345
        exp_seq = skbio.DNA('AATTGGCCTTAAATAT')
        exp_breaks = np.array([0, 6, 12, 16])
        obs_seq, obs_breaks = join_contigs(sequences)
        self.assertEqual(obs_seq, exp_seq)
        npt.assert_equal(obs_breaks, exp_breaks)

    def remap(self, seq_as_array):
        d = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}
        return ''.join((d[c] for c in seq_as_array))

    def test_sample_from_contig_set_one_short(self):
        randfunc = np.random.randint

        np.random.seed(1234)
        # An integration test
        #                       0123456789012345678901234567890123456789
        sequences = [skbio.DNA('ATGCAATTGGCCAAATTTGGGCCCAAAATTTTGGGGCCCC'),
                     skbio.DNA('CGTACCGGTT')]
        fullseq = skbio.DNA.concat(sequences)

        depth = 100
        length = 15

        obs = sample_from_contig_set(sequences, depth, length, randfunc)

    def test_sample_from_contig_set(self):
        def mock(start, stop, n):
            if start == 0:
                return np.tile([0, 5, 10, 15, 9, 12], 100)[:n]
            else:
                return np.tile([40, 41, 42, 43], 100)[:n]

        np.random.seed(1234)
        # An integration test
        #                       0123456789012345678901234567890123456789
        sequences = [skbio.DNA('ATGCAATTGGCCAAATTTGGGCCCAAAATTTTGGGGCCCC'),
                     skbio.DNA('CGTACCGGTT')]
        fullseq = skbio.DNA.concat(sequences)

        depth = 100
        length = 3

        obs = sample_from_contig_set(sequences, depth, length, mock)

        indices = []
        for o in obs:
            remapped = self.remap(o)
            self.assertIn(remapped, fullseq)
            indices.append(fullseq.index(remapped))

        # we expect the both the first and second sequence to be fully
        # represented by our starting indices except in rare stochastic
        # scenario (as on average, 20 reads will come from the second contig)
        self.assertTrue(set(indices) == {0, 5, 10, 15, 9, 12, 40, 41, 42, 43})

        # we could potentially verify multinomial is working as expected but
        # that may be getting a bit pedantic.


if __name__ == '__main__':
    unittest.main()
