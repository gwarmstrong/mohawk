import unittest
from unittest import mock
import pandas as pd
from io import BufferedReader
from mohawk.data_downloader import (_get_ids_not_downloaded,
                                    get_ftp_dir,
                                    representative_genomes_file,
                                    complete_genomes_file
                                    )


class TestResources(unittest.TestCase):

    def test_finds_representative_ftp(self):
        ftp_links = representative_genomes_file()
        self.assertIsInstance(ftp_links, BufferedReader)

    def test_finds_complete_ftp(self):
        ftp_links = complete_genomes_file()
        self.assertIsInstance(ftp_links, BufferedReader)


class TestGetFTPDir(unittest.TestCase):

    def test_get_ftp_dir(self):
        abspath = 'ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/900/128/' \
                  '725/GCF_900128725.1_BCifornacula_v1.0'
        expected_dir = '/genomes/all/GCF/900/128/725/GCF_900128725' \
                       '.1_BCifornacula_v1.0'
        observed_dir = get_ftp_dir(abspath)
        self.assertEqual(observed_dir, expected_dir)


class TestGetIdsNotDownLoaded(unittest.TestCase):

    def setUp(self) -> None:
        self.id_list = ['ExpA', 'ExpB']
        self.genomes_metadata = pd.DataFrame({'ftp_path': [
            'ftp://path/to/ftp/id_A', 'ftp://path/to/ftp/id_B'],
            'fna_gz_name': ['id_A_genomic.fna.gz', 'id_B_genomic.fna.gz']},
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

