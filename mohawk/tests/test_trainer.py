from mohawk.tests import testing
from mohawk.trainer import trainer
from mohawk.models import ConvNetAvg, ConvNetAvg2


class TestTrainer(testing.MohawkTestCase):

    package = 'mohawk.tests'

    def setUp(self) -> None:
        super(TestTrainer, self).setUp()
        self.distribution = [0.5, 0.3, 0.2]
        self.id_list = [
                        'GCF_001999985.1',  # Helicobacter bilis
                        'GCF_002209165.2',  # Staphylococcus sciuri
                        'GCF_000953655.1',  # Legionella hackeliae
        ]
        self.classes = [
                        'class1',
                        'class2'
                        'class2'
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
                        'log_dir': self.create_data_path('model_log_1')}
        mod = trainer(ConvNetAvg, self.distribution, self.total_reads,
                      self.length,
                      self.train_ratio, self.id_list,
                      data_directory=self.get_data_path('sample_genomes'),
                      train_kwargs=train_kwargs)

        self.assertEqual(mod.n_classes, 3)

    def test_trainer_runs_with_classes(self):
        train_kwargs = {'gpu': False,
                        'summary_interval': 1,
                        'epochs': 10,
                        'summarize': True,
                        'learning_rate': 0.001,
                        'log_dir': self.create_data_path('model_log_2')}
        mod = trainer(ConvNetAvg2, self.distribution, self.total_reads,
                      self.length,
                      self.train_ratio, self.id_list, class_list=self.classes,
                      data_directory=self.get_data_path('sample_genomes'),
                      train_kwargs=train_kwargs)

        self.assertEqual(mod.n_classes, 2)
