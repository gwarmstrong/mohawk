import unittest
from unittest import mock
import pandas as pd
from mohawk.data_generation import _get_ids_not_downloaded


class TestGetIdsNotDownLoaded(unittest.TestCase):

    def setUp(self) -> None:
        self.id_list = ['ExpA', 'ExpB']
        self.genomes_metadata = pd.DataFrame({'ftp_path': [
            'ftp://path/to/ftp/id_A', 'ftp://path/to/ftp/id_B']},
            index=self.id_list)
        self.genomes_directory = '/some/path/to/dir'

    def test_get_ids_simple(self):
        with mock.patch('os.listdir') as mocked_listdir:
            mocked_listdir.return_value = ['id_B_genomic.fna.gz']
            returned_ids = _get_ids_not_downloaded(self.id_list,
                                                   self.genomes_metadata,
                                                   self.genomes_directory)
        self.assertCountEqual(returned_ids, ['ExpA'])

    def test_has_fna(self):
        with mock.patch('os.listdir') as mocked_listdir:
            mocked_listdir.return_value = ['id_A_genomic.fna.gz',
                                           'id_B_genomic.fna']
            returned_ids = _get_ids_not_downloaded(self.id_list,
                                                   self.genomes_metadata,
                                                   self.genomes_directory)
        self.assertCountEqual(returned_ids, [])

    def test_get_ids_simple_fna_only_gz_present(self):
        with mock.patch('os.listdir') as mocked_listdir:
            mocked_listdir.return_value = ['id_B_genomic.fna.gz']
            returned_ids = _get_ids_not_downloaded(['ExpB'],
                                                   self.genomes_metadata,
                                                   self.genomes_directory,
                                                   fna_only=True)
        self.assertCountEqual(returned_ids, ['ExpB'])

    def test_get_ids_simple_fna_only_fna_present(self):
        with mock.patch('os.listdir') as mocked_listdir:
            mocked_listdir.return_value = ['id_A_genomic.fna']
            returned_ids = _get_ids_not_downloaded(['ExpA'],
                                                   self.genomes_metadata,
                                                   self.genomes_directory,
                                                   fna_only=True)
        self.assertCountEqual(returned_ids, [])

    def test_get_ids_simple_fna_only_neither_present(self):
        with mock.patch('os.listdir') as mocked_listdir:
            mocked_listdir.return_value = []
            with self.assertRaisesRegex(ValueError, r'Cannot gunzip when .gz '
                                                    r'file is not present'):
                _get_ids_not_downloaded(['ExpA'],
                                        self.genomes_metadata,
                                        self.genomes_directory,
                                        fna_only=True)


