import numpy as np
import matplotlib.pyplot as plt
from callbacks import Callback

class BinaryAccuracyCallback(Callback):
    
    def on_train_begin(self):
        self.state_dict['accs'] = []
        
    def on_epoch_begin(self):
        self.weighted_values = []
        self.num_examples = 0
        
    def eval_on_forward_end(self):
        
        preds = self.state_dict['yhatb'].detach().numpy()
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0
        preds = preds.astype(int)
        
        targets = self.state_dict['yb'].numpy()
        targets = targets.astype(int)
        
        acc = np.mean(preds == targets)

        self.weighted_values.append(acc * self.state_dict['bs'])
        self.num_examples += self.state_dict['bs']
        
    def on_epoch_end(self):
        self.state_dict['accs'].append(np.sum(self.weighted_values) / self.num_examples)
        
    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.state_dict['accs'])
        ax.set_ylim(0.5, 1)
        ax.grid()
        ax.set_xlabel('Epoch'); ax.set_ylabel('Test acc')
        
class MulticlassAccuracyCallback(Callback):
    
    def on_train_begin(self):
        self.state_dict['accs'] = []
        
    def on_epoch_begin(self):
        self.weighted_values = []
        self.num_examples = 0
        
    def eval_on_forward_end(self):
        
        preds = self.state_dict['yhatb'].detach().numpy()
        preds = preds.argmax(axis=1)
        preds = preds.astype(int)
        
        targets = self.state_dict['yb'].numpy()
        targets = targets.astype(int)
        
        acc = np.mean(preds == targets)

        self.weighted_values.append(acc * self.state_dict['bs'])
        self.num_examples += self.state_dict['bs']
        
    def on_epoch_end(self):
        self.state_dict['accs'].append(np.sum(self.weighted_values) / self.num_examples)
        
    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.state_dict['accs'])
        ax.set_ylim(0.5, 1)
        ax.grid()
        ax.set_xlabel('Epoch'); ax.set_ylabel('Test acc')
        
class LossCallback(Callback):
    
    def on_train_begin(self):
        self.state_dict['losses'] = []
        
    def on_epoch_begin(self):
        self.weighted_values = []
        self.num_examples = 0
        
    def eval_on_loss_end(self):
        self.weighted_values.append(self.state_dict['lossb'] * self.state_dict['bs'])
        self.num_examples += self.state_dict['bs']
        
    def on_epoch_end(self):
        self.state_dict['losses'].append(np.sum(self.weighted_values) / self.num_examples)
        
    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(self.state_dict['losses'])
        ax.grid()
        ax.set_xlabel('Epoch'); ax.set_ylabel('Test loss')