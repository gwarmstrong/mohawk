import unittest
import pkg_resources
import os


class MohawkTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.to_remove = []

    def get_data_path(self, filename):
        # adapted from qiime2.plugin.testing.TestPluginBase and biocore/unifrac
        return pkg_resources.resource_filename(self.package,
                                               'data/%s' % filename)

    def create_data_path(self, filename):
        path = self.get_data_path(filename)
        self.to_remove.append(path)
        return path

    def tearDown(self) -> None:
        for file_ in self.to_remove:
            os.remove(file_)
