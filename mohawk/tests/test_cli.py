from mohawk.tests import testing
from mohawk.scripts import mohawk as mohawk_cli
from click.testing import CliRunner


class TestCLI(testing.MohawkTestCase):

    package = 'mohawk.tests'

    def setUp(self) -> None:
        super(TestCLI, self).setUp()

    def test_cli_basic(self):
        runner = CliRunner()
        id_path = self.get_data_path('sample_genome_ids.tsv')
        data_dir = self.get_data_path('sample_genomes')
        hparams = self.get_data_path('additional_hparams.txt')
        log_dir = self.create_data_path('cli_model_log')
        train_args = ['--model-name', 'ConvNetAvg',
                      '--genome-ids', id_path,
                      '--log-dir', log_dir,
                      '--data-dir', data_dir,
                      '--additional-hyper-parameters', hparams
                      ]
        result = runner.invoke(
            mohawk_cli.seq_by_seq_pytorch,
            args=train_args)
        self.assertEqual(result.exit_code, 0)
