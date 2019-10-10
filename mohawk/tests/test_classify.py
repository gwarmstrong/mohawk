from mohawk.classify import ensure_lengths, classify
from mohawk.tests import testing


class TestClassifyHelpers(testing.MohawkTestCase):

    def test_ensure_lengths(self):
        length = 3
        seq1 = [0, 1, 2]
        seq2 = [0, 2, 1]
        seq3 = [0, 2, 2]
        ids = ['seq1', 'seq2', 'seq3']
        input_seqs = [seq1, seq2, seq3]
        seqs, ids = ensure_lengths(list(zip(input_seqs, ids)), length=length)

        expected_seqs = [[0, 1, 2],
                         [0, 2, 1],
                         [0, 2, 2]]
        expected_ids = ['seq1',
                        'seq2',
                        'seq3']
        list_seqs = [[l1 for l1 in l2] for l2 in seqs]
        self.assertListEqual(list_seqs, expected_seqs)
        self.assertListEqual(ids, expected_ids)


class TestClassify(testing.MohawkTestCase):

    # makes `get_data_path` find appropriate path
    package = 'mohawk.tests'

    def test_classify_runs(self):
        model = self.get_data_path('test_model.mod')
        sequence_file = self.get_data_path('small_test_fq.fq')
        length = 150
        format = 'fastq'
        format_kwargs = {'phred_offset': 33}
        classify(model, length, sequence_file,
                 batch_size=640,
                 format_=format,
                 format_kwargs=format_kwargs)
