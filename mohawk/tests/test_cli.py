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
        log_dir = self.create_data_path('cli_model_log')
        train_args = ['--model-name', 'ConvNetAvg',
                      '--genome-ids', id_path,
                      '--log-dir', log_dir,
                      '--data-dir', data_dir
                      ]
        result = runner.invoke(mohawk_cli.train, args=train_args)
        print(result.output)
        print(result.exc_info)
        import traceback
        traceback.print_tb(result.exc_info[2])
        print(result.exception)
        self.assertEqual(result.exit_code, 0)
