import unittest
from io import BufferedReader

from mohawk.utils import representative_genomes_file, complete_genomes_file


class TestResources(unittest.TestCase):

    def test_finds_representative_ftp(self):
        ftp_links = representative_genomes_file()
        self.assertIsInstance(ftp_links, BufferedReader)

    def test_finds_complete_ftp(self):
        ftp_links = complete_genomes_file()
        self.assertIsInstance(ftp_links, BufferedReader)
