from mohawk.tests import testing
from mohawk.trainer import trainer
from mohawk.models import ConvNetAvg


class TestTrainer(testing.MohawkTestCase):

    package = 'mohawk.tests'

    def setUp(self) -> None:
        super(TestTrainer, self).setUp()
        self.model = ConvNetAvg
        self.distribution = [0.5, 0.5]
        self.id_list = [
                        'GCF_001999985.1',  # Helicobacter bilis
                        'GCF_002209165.2',  # Staphylococcus sciuri
                        ]
        self.total_reads = 200
        self.length = 150
        self.train_ratio = 0.8

    def test_trainer_runs(self):
        train_kwargs = {'gpu': False,
                        'summary_interval': 1,
                        'epochs': 10,
                        'summarize': True,
                        'learning_rate': 0.001,
                        'log_dir': self.get_data_path('model_log_1')}
        model = trainer(self.model, self.distribution, self.total_reads,
                        self.length, self.train_ratio, self.id_list,
                        data_directory=self.get_data_path('demo_data_dir_1'),
                        train_kwargs=train_kwargs)
