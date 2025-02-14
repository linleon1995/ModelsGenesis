import os
import torch
import numpy as np
from tensorboardX import SummaryWriter
import sys

from modules.utils import metrics


class Trainer(object):
    def __init__(self,
                 model, 
                 criterion, 
                 optimizer, 
                 train_dataloader, 
                 valid_dataloader,
                 logger,
                 device,
                 n_class,
                 exp_path,
                 train_epoch=10,
                 batch_size=1,
                 valid_activation=None,
                 USE_TENSORBOARD=True,
                 USE_CUDA=True,
                 history=None,
                 checkpoint_saving_steps=20
                 ):
        self.n_class = n_class
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.logger = logger
        self.iterations = 0
        self.device = device
        self.history = history
        self.model = model
        if history is not None:
            self.load_model_from_checkpoint(self.history)
        self.valid_activation = valid_activation
        self.USE_TENSORBOARD = USE_TENSORBOARD
        self.exp_path = exp_path
        if self.USE_TENSORBOARD:
            self.train_writer = SummaryWriter(log_dir=os.path.join(self.exp_path, 'train'))
            self.valid_writer = SummaryWriter(log_dir=os.path.join(self.exp_path, 'valid'))
        if USE_CUDA:
            self.model.cuda()
        self.max_acc = 0
        train_samples = len(self.train_dataloader.dataset)
        self.display_step = self.calculate_display_step(num_sample=train_samples, batch_size=self.batch_size)
        self.checkpoint_saving_steps = checkpoint_saving_steps

    def fit(self):
        for self.epoch in range(1, self.train_epoch + 1):
            self.train()
            with torch.no_grad():
                if self.valid_dataloader is not None:
                    self.validate()
                self.save_model()

        if self.USE_TENSORBOARD:
            self.train_writer.close()
            self.valid_writer.close()
            
    def train(self):
        self.model.train()
        print(60*"=")
        self.logger.info(f'Epoch {self.epoch}/{self.train_epoch}')
        train_samples = len(self.train_dataloader.dataset)
        total_train_loss = 0.0
        for i, data in enumerate(self.train_dataloader, self.iterations + 1):
            input_var, target_var = data['input'], data['target']
            # TODO input type
            input_var = input_var.float()
            target_var = target_var.long()
            input_var, target_var = input_var.to(self.device), target_var.to(self.device)

            # TODO: what is aux in model?
            batch_output = self.model(input_var)['out']
            
            self.optimizer.zero_grad()
            loss = self.criterion(batch_output, target_var)
            loss.backward()
            self.optimizer.step()
        
            loss = loss.item()
            total_train_loss += loss*input_var.shape[0]
            if self.USE_TENSORBOARD:
                self.train_writer.add_scalar('Loss/step', loss, i)

            if i%self.display_step == 0:
                self.logger.info('Step {}  Step loss {}'.format(i, loss))
        self.iterations = i
        if self.USE_TENSORBOARD:
            # TODO: correct total_train_loss
            self.train_writer.add_scalar('Loss/epoch', total_train_loss/train_samples, self.epoch)

    def validate(self):
        self.model.eval()
        self.eval_tool = metrics.SegmentationMetrics(self.n_class, ['accuracy'])
        test_n_iter, total_test_loss = 0, 0
        valid_samples = len(self.valid_dataloader.dataset)
        for idx, data in enumerate(self.valid_dataloader):
            test_n_iter += 1

            input_var, labels = data['input'], data['target']
            # TODO input type
            input_var = input_var.float()
            labels = labels.long()


            input_var, labels = input_var.to(self.device), labels.to(self.device)
            # TODO: what is aux in model?
            outputs = self.model(input_var)['out']

            loss = self.criterion(outputs, labels)

            # loss = loss_func(outputs, torch.argmax(labels, dim=1)).item()
            total_test_loss += loss.item()

            prob = self.valid_activation(outputs)
            prediction = torch.argmax(prob, dim=1)
            # TODO:
            labels = labels[:,1]

            labels = labels.cpu().detach().numpy()
            prediction = prediction.cpu().detach().numpy()

            evals = self.eval_tool(labels, prediction)

        self.avg_test_acc = metrics.accuracy(
                np.sum(self.eval_tool.total_tp), np.sum(self.eval_tool.total_fp), np.sum(self.eval_tool.total_fn), np.sum(self.eval_tool.total_tn)).item()
        self.valid_writer.add_scalar('Accuracy/epoch', self.avg_test_acc, self.epoch)
        self.valid_writer.add_scalar('Loss/epoch', total_test_loss/valid_samples, self.epoch)

    def load_model_from_checkpoint(self, ckpt, model_state_key='model_state_dict'):
        state_key = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(state_key[model_state_key])
        self.model = self.model.to(self.device)

    def save_model(self):
        checkpoint = {
            "net": self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "epoch": self.epoch
            }
        if self.avg_test_acc > self.max_acc:
            self.max_acc = self.avg_test_acc
            self.logger.info(f"-- Saving best model with testing accuracy {self.max_acc:.3f} --")
            checkpoint_name = 'ckpt_best.pth'
            torch.save(checkpoint, os.path.join(self.exp_path, checkpoint_name))

        # if self.epoch%self.config.TRAIN.CHECKPOINT_SAVING_STEPS == 0:
        if self.epoch%20 == 0:
            self.logger.info(f"Saving model with testing accuracy {self.avg_test_acc:.3f} in epoch {self.epoch} ")
            checkpoint_name = 'ckpt_best_{:04d}.pth'.format(self.epoch)
            torch.save(checkpoint, os.path.join(self.exp_path, checkpoint_name))


    def calculate_display_step(self, num_sample, batch_size, display_times=5):
        num_steps = max(num_sample//batch_size, 1)
        display_steps = max(num_steps//display_times, 1)
        # display_steps = max(num_steps//display_times//display_times*display_times, 1)
        return display_steps


if __name__ == '__main__':
    pass