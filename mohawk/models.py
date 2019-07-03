import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
from sklearn.metrics import f1_score
import numpy as np


class BaseModel(nn.Module):

    def __init__(self, seed=None, **kwargs):
        super(BaseModel, self).__init__()
        if seed is not None:
            torch.manual_seed(seed+2)

    def forward(self, data):
        pass

    # might need to separate fit's kwargs from trainer's kwargs
    def fit(self,
            train_dataset: DataLoader,
            val_dataset: Optional[DataLoader],
            seed: Optional[int],
            log_dir: Optional[str],
            epochs: Optional[int],
            **kwargs):
        pass

    def forward_step(self, data):
        x = data['read']
        x = x.to(self.device)
        y_pred = self(x)
        y = data['label']
        y = y.to(self.device)
        return x, y, y_pred

    def backward_step(self, y, y_pred, optimizer):
        loss = self.loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def accuracy(self, dataloader, cutoff=0.):
        # make sure to zero grad after running ? -> do we need to?
        total_correct = 0
        total_present = 0
        for data in dataloader:
            _, y, y_pred = self.forward_step(data)
            y_pred_max = y_pred.max(1)
            y_pred_class = y_pred_max.indices
            y_pred_max_val = y_pred_max.values
            over_cutoff = y_pred_max_val > cutoff
            correct = torch.eq(y, y_pred_class) & over_cutoff
            total_correct += correct.sum().item()
            total_present += over_cutoff.sum().item()

        # correct for no predictions made
        if total_present == 0:
            total_present = 1

        return total_correct / total_present

    def avg_loss(self, dataloader):
        total_loss = 0
        total_samples = 0
        for data in dataloader:
            _, y, y_pred = self.forward_step(data)
            # use this and not backward step so weights are not updated
            loss = self.loss_fn(y_pred, y)
            total_loss += loss.item()
            total_samples += len(y)

        return total_loss / total_samples

    def f1_score_(self, dataloader, average=None):
        all_class_predictions = []
        all_class_labels = []
        for data in dataloader:
            _, y, y_pred = self.forward_step(data)
            y_pred_class = y_pred.argmax(1)
            all_class_predictions.append(y_pred_class.cpu())
            all_class_labels.append(y)

        predictions = torch.cat(all_class_predictions, 0).cpu()
        labels = torch.cat(all_class_labels, 0).cpu()

        return f1_score(labels, predictions, average=average)

    def max_softmax(self, dataloader, split=False):
        all_max_preds = []
        all_is_correct = []  # TODO rename
        max_pred_class = []
        for data in dataloader:
            _, y, y_pred = self.forward_step(data)
            y_pred_max = y_pred.max(1)
            all_max_preds.append(y_pred_max.values)
            if split:
                max_pred_class.append(y_pred_max.indices)
                all_is_correct.append(y)

        max_preds = torch.cat(all_max_preds, 0)
        if split:
            max_class = torch.cat(max_pred_class, 0)
            correct_class = torch.cat(all_is_correct, 0)

            # 1 if match, 0 otherwise
            correct = torch.eq(max_class, correct_class)

            correct = correct.cpu().detach().numpy()
            max_preds = max_preds.cpu().detach().numpy()
            correct_max = max_preds[np.where(correct == 1)]
            incorrect_max = max_preds[np.where(correct == 0)]
            return correct_max if len(correct_max) > 0 else None, \
                incorrect_max if len(incorrect_max) > 0 else None

        else:
            return max_preds

    def concise_summary(self, dataloader):
        total_correct = 0
        total_present = 0
        total_loss = 0
        total_samples = 0
        for data in dataloader:
            _, y, y_pred = self.forward_step(data)
            y_pred_max = y_pred.max(1)
            y_pred_class = y_pred_max.indices
            y_pred_max_val = y_pred_max.values
            over_cutoff = y_pred_max_val > 0
            correct = torch.eq(y, y_pred_class) & over_cutoff
            total_correct += correct.sum().item()
            total_present += over_cutoff.sum().item()
            loss = self.loss_fn(y_pred, y)
            total_loss += loss.item()
            total_samples += len(y)

        # correct for no predictions made
        if total_present == 0:
            total_present = 1

        return {'accuracy-global': total_correct / total_present,
                'avg-loss': total_loss / total_samples
                }

    def fit(self,
            train_dataset: DataLoader,
            val_dataset: Optional[DataLoader] = None,
            external_dataset: Optional[DataLoader] = None,
            seed: Optional[int] = None,
            log_dir: Optional[str] = None,
            epochs: int = 1000,
            summary_interval: int = 10,
            learning_rate: float = 0.0001,
            gpu: bool = False,
            summarize: bool = True,
            summary_kwargs: Optional[dict] = dict(),
            **kwargs):

        if seed is not None:
            torch.manual_seed(seed+3)

        writer = SummaryWriter(log_dir=log_dir)

        optimizer = self.optim(self.parameters(),
                               lr=learning_rate,
                               weight_decay=0)

        self.device = torch.device('cuda' if gpu else 'cpu')
        # self.to(device)

        # put computational graph in tensorboard TODO declutter
        for batch_index, data in enumerate(train_dataset):
            if batch_index == 0:
                writer.add_graph(self,
                                 input_to_model=data['read'].to(self.device)
                                 )

        # run for an extra epoch to hit a multiple of 10 # TODO should go?
        for index_epoch in range(epochs + 1):
            for data in train_dataset:
                _, y, y_pred = self.forward_step(data)
                self.backward_step(y, y_pred, optimizer)

            if summarize and (index_epoch % summary_interval == 0):
                self.summarize(writer,
                               counter=index_epoch,
                               train_dataset=train_dataset,
                               val_dataset=val_dataset,
                               external_dataset=external_dataset,
                               **summary_kwargs)

        writer.close()

    def summarize(self,
                  writer,
                  counter=0,
                  average='weighted',
                  concise=True,
                  train_dataset=None,
                  val_dataset=None,
                  external_dataset=None,
                  classify_threshold=None):

        datasets = {'train': train_dataset,
                    'val': val_dataset,
                    'ext': external_dataset}
        datasets = {name: dataset for name, dataset in datasets.items() if
                    dataset is not None}
        if concise:
            summary_stats = self.summary_helper(datasets, self.concise_summary)
            extract_metrics = ['accuracy-global', 'avg-loss']
            reshaped_stats = dict()
            for metric in extract_metrics:
                reshaped_stats[metric] = {dataset: summary_stats[dataset][
                    metric] for dataset in summary_stats}

                writer.add_scalars(metric,
                                   reshaped_stats[metric],
                                   counter)
            for name, acc in reshaped_stats['accuracy-global'].items():
                print("IT: {}, {} ACC: {}".format(counter,
                                                  name,
                                                  acc))

        else:
            # TODO could have self.add_scalar and self.add_histogram methods ?
            accuracies = self.summary_helper(datasets, self.accuracy)
            writer.add_scalars('accuracy-global', accuracies, counter)

            if classify_threshold is not None:
                threshold_accuracies = self.summary_helper(
                    datasets, self.accuracy, {'cutoff': classify_threshold})
                writer.add_scalars('accuracy-threshold_{}'.format(
                    classify_threshold),
                    threshold_accuracies,
                    counter)

            avg_losses = self.summary_helper(datasets, self.avg_loss)
            writer.add_scalars('loss',
                               avg_losses,
                               counter)

            f1_scores = self.summary_helper(datasets, self.f1_score_,
                                            {'average': average})
            writer.add_scalars('f1-score',
                               f1_scores,
                               counter)

            max_softmax = self.summary_helper(datasets, self.max_softmax)
            # writing this one is weird
            for name, softmax in max_softmax.items():
                writer.add_histogram('max_softmax/{}/all'.format(name),
                                     softmax,
                                     counter)

            max_softmax_split = self.summary_helper(datasets,
                                                    self.max_softmax,
                                                    {'split': True})
            # writing this one is weird
            for name, (correct, incorrect) in max_softmax_split.items():
                if correct is not None:
                    writer.add_histogram('max_softmax/{}/correct'.format(name),
                                         correct,
                                         counter)
                if incorrect is not None:
                    writer.add_histogram('max_softmax/{}/incorrect'.format(
                        name),
                        incorrect,
                        counter)

            for name, acc in accuracies.items():
                print("IT: {}, {} ACC: {}".format(counter,
                                                  name,
                                                  acc))

    @staticmethod
    def summary_helper(datasets, metric, metric_kwargs=dict()) -> dict:
        metrics = {name: metric(dataset, **metric_kwargs) for name, dataset in
                   datasets.items()}
        return {name: value for name, value in metrics.items() if value is
                not None}


class SmallConvNet(BaseModel):

    def __init__(self,
                 n_classes: int,
                 length: int,
                 seed: Optional[int] = None,
                 ):
        super(SmallConvNet, self).__init__(seed=seed)

        self.conv1_kernel_size = 7
        self.n_bases = 4
        self.n_filters = 11
        self.seq_length = length
        self.n_classes = n_classes
        self.conv1 = nn.Conv1d(self.n_bases, self.n_filters,
                               kernel_size=self.conv1_kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1_in_dim = int((self.seq_length - self.conv1_kernel_size + 1)
                              / 2) * self.n_filters

        self.fc1 = nn.Linear(self.fc1_in_dim, 50)
        self.fc2 = nn.Linear(50, self.n_classes)
        self.softmax = nn.Softmax(dim=1)
        self.optim = Adam
        self.loss_fn = CrossEntropyLoss(reduction='sum')
        self.device = None

    def forward(self, data):

        x = self.conv1(data)

        x = F.relu(x)

        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)

        x = F.relu(x)

        x = self.fc2(x)

        x = self.softmax(x)

        return x


class ConvNet2(BaseModel):
    def __init__(self,
                 n_classes: int,
                 length: int,
                 seed: Optional[int] = None,
                 ):
        super(ConvNet2, self).__init__(seed=seed)

        self.loss_fn = CrossEntropyLoss(reduction='sum')
        self.optim = Adam

        dilations = [1, 2, 4, 8, 16]
        channels = [4, 8, 16, 8, 4, 1]
        linear_sizes = [129, 200, 100, 50, n_classes]  # TODO how 119?
        self.conv = nn.Sequential()
        for i, d in enumerate(dilations):
            self.conv.add_module('Conv_' + str(i),
                                 nn.Conv1d(in_channels=channels[i],
                                           out_channels=channels[i + 1],
                                           kernel_size=9,
                                           dilation=d
                                           )
                                 )
            self.conv.add_module('Conv_' + str(i)+'_relu', nn.ReLU())

        self.fc = nn.Sequential()
        for i in range(1, len(linear_sizes)):
            self.fc.add_module('FC_' + str(i),
                                 nn.Linear(linear_sizes[i - 1],
                                           linear_sizes[i]),
                                 )
            if i < len(linear_sizes) - 1:
                self.fc.add_module('FC_' + str(i) + '_relu', nn.ReLU())
            else:
                self.fc.add_module('Softmax', nn.Softmax(dim=1))

    def forward(self, data):
        x = self.conv(data)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConvNetAvg(BaseModel):
    def __init__(self,
                 n_classes: int,
                 length: int,
                 seed: Optional[int] = None,
                 ):
        super(ConvNetAvg, self).__init__(seed=seed)

        self.loss_fn = CrossEntropyLoss(reduction='sum')
        self.optim = Adam

        dilations = [1, 2, 4, 8, 16]
        channels = [4, 20, 40, 80, 100, 120] # first has to be 4
        linear_sizes = [channels[-1], 200, 100, 50, n_classes]  # TODO how 119?
        self.conv = nn.Sequential()
        for i, d in enumerate(dilations):
            self.conv.add_module('Conv_' + str(i),
                                 nn.Conv1d(in_channels=channels[i],
                                           out_channels=channels[i + 1],
                                           kernel_size=5,
                                           dilation=d
                                           )
                                 )
            self.conv.add_module('Conv_' + str(i)+'_relu', nn.ReLU())

        self.fc = nn.Sequential()
        for i in range(1, len(linear_sizes)):
            self.fc.add_module('FC_' + str(i),
                               nn.Linear(linear_sizes[i - 1],
                                         linear_sizes[i]),
                               )
            if i < len(linear_sizes) - 1:
                self.fc.add_module('FC_' + str(i) + '_relu', nn.ReLU())
            else:
                self.fc.add_module('Softmax', nn.Softmax(dim=1))

    def forward(self, data):
        x = self.conv(data)
        x = x.mean(-1)
        x = self.fc(x)
        return x
