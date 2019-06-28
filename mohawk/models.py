import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss, CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from typing import Optional


class BaseModel(nn.Module):

    def __init__(self, seed=None, **kwargs):
        super(BaseModel, self).__init__()
        if seed is not None:
            torch.manual_seed(seed+2)

    def forward(self, data):
        pass

    def summarize(self, writer, **kwargs):
        # TODO train and test accuracies
        # maybe save predictions?
        # definitely save trained model
        # TODO tensorboard logs?
        # from torch.utils import tensorboard
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

    def accuracy(self, dataloader):
        # make sure to zero grad after running ? -> do we need to?
        total_correct = 0
        total_present = 0
        for data in dataloader:
            x = data['read']
            y_pred = self.forward(x)
            y_pred_class = y_pred.argmax(1)
            correct = torch.eq(data['label'], y_pred_class)
            total_correct += correct.sum()
            total_present += len(correct)

        return total_correct.item() / total_present

    def avg_loss(self, dataloader):
        total_loss = 0
        total_samples = 0
        for data in dataloader:
            x = data['read']
            y_pred = self.forward(x)
            y = data['label']
            loss = self.loss_fn(y_pred, y)
            total_loss += loss.item()
            total_samples += len(y)

        return total_loss / total_samples


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

    def fit(self,
            train_dataset: DataLoader,
            val_dataset: Optional[DataLoader],
            seed: Optional[int],
            log_dir: Optional[str],
            epochs: int = 10000,
            gpu: bool = False,
            **kwargs):
        if seed is not None:
            torch.manual_seed(seed+3)

        writer = SummaryWriter(log_dir=log_dir)

        optimizer = self.optim(self.parameters(), lr=0.0001, weight_decay=0)

        if gpu:
            self.to('cuda')

        # give hint to where model is being trained (see if moved to gpu)
        print("Training on: {}".format(self.fc1.weight.device()))

        for index_epoch in range(epochs):
            loss_epoch = 0
            for data in train_dataset:
                # print(data['read'])
                # print(data['read'].shape)
                x = data['read']
                if gpu:
                    x.to('cuda')
                y_pred = self.forward(x)
                # loss on data['label']
                y = data['label']
                if gpu:
                    y.to('cuda')
                loss = self.loss_fn(y_pred, y)
                loss_epoch += loss.item()
                optimizer.zero_grad()
                # loss.backward()
                loss.backward()
                # other maintenance
                optimizer.step()

            if index_epoch % 10 == 0:
                self.summarize(writer, index_epoch, train_dataset, val_dataset)

        writer.close()

    def summarize(self,
                  writer,
                  counter=0,
                  train_dataset=None,
                  val_dataset=None):

        train_accuracy = self.accuracy(train_dataset)
        val_accuracy = self.accuracy(val_dataset)
        writer.add_scalars('accuracy',
                           {'train': train_accuracy,
                            'val': val_accuracy},
                           counter)

        train_avg_loss = self.avg_loss(train_dataset)
        val_avg_loss = self.avg_loss(val_dataset)
        writer.add_scalars('loss',
                           {'train': train_avg_loss,
                            'val': val_avg_loss},
                           counter)
        # writer.add_scalar('val_accuracy', val_accuracy, counter)
        print("IT: {}, Train ACC: {}".format(counter,
                                             train_accuracy))
        print("IT: {}, Val ACC: {}".format(counter,
                                           val_accuracy))

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




