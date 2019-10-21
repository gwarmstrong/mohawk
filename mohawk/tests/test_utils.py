import unittest
from io import BufferedReader
import pandas as pd

from mohawk.utils import default_metadata, _ftp_path, get_zipped_fasta_name,\
        gz_stripper, full_fna_path


class TestResources(unittest.TestCase):

    def test_finds_default_metadata(self):
        metadata = default_metadata()
        self.assertIsInstance(metadata, BufferedReader)


class TestResourceHelpers(unittest.TestCase):

    def setUp(self) -> None:
        metadata = default_metadata()
        self.metadata = pd.read_csv(metadata, sep='\t', index_col=0)

    def test_ftp_path(self):
        id_ = 'GCF_000001765.3'
        exp = 'ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/765' \
              '/GCF_000001765.3_Dpse_3.0'
        self.assertEqual(_ftp_path(id_, self.metadata), exp)

    def test_get_zipped_fasta_name(self):
        id_ = 'GCF_000001765.3'
        exp = 'GCF_000001765.3_Dpse_3.0_genomic.fna.gz'
        self.assertEqual(get_zipped_fasta_name(id_, self.metadata), exp)

    def test_gz_stripper(self):
        input_ = 'GCF_000001765.3_Dpse_3.0_genomic.fna.gz'
        exp = 'GCF_000001765.3_Dpse_3.0_genomic.fna'
        self.assertEqual(gz_stripper(input_), exp)
        self.assertEqual(gz_stripper(exp), exp)

    def test_full_fna_path(self):
        seq_dir = '/path/to/dir'
        id_ = 'GCF_000001765.3'
        exp = '/path/to/dir/GCF_000001765.3/' \
              'GCF_000001765.3_Dpse_3.0_genomic.fna'
        self.assertEqual(full_fna_path(seq_dir, id_, self.metadata), exp)

