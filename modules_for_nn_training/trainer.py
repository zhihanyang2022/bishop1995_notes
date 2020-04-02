import numpy as np
from tqdm import tqdm_notebook
import matplotlib
import matplotlib.pyplot as plt
import torch

class Trainer:
    
    def __init__(self, learn, cb_handler):
        self.learn = learn
        self.cb_handler = cb_handler
        
    @staticmethod
    def plot_lrs_and_losses(lrs, losses, lr_axis_min, lr_axis_max, show_suggestion):
        
        plt.plot(lrs, losses)
        
        if show_suggestion:  # does not work unless losses are rather smooth
            min_grad_idx = np.argmin(np.gradient(losses))
            plt.scatter(lrs[min_grad_idx], losses[min_grad_idx], color='red')
        
        plt.grid(which='major'); plt.grid(which='minor', axis='x')
        plt.xscale('log')
    
        # base 10 ticks is easier to read off of
        xticks = []
        value = lr_axis_min
        while value <= lr_axis_max:
            xticks.append(value)
            value *= 10
            
        plt.xticks(xticks)
        # reference: https://github.com/fastai/fastai/blob/3b7c453cfa3845c6ffc496dd4043c07f3919270e/fastai/basic_train.py#L540
        plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        
        plt.xlabel('Learning rate'); plt.ylabel('Loss')
        plt.show()
        
    def find_lr(self, init_value=1e-6, final_value=10, num_itr=200, beta=0.98, skip_start=10, skip_end=5, show_suggestion=False):
        
        mult = (final_value / init_value) ** (1 / num_itr)
        lr = init_value
        
        for param_group in self.learn.opt.param_groups:
            param_group['lr'] = lr
        
        avg_loss = 0.
        best_loss = np.inf
        batch_num = 0
        losses = []
        lrs = []
        
        num_epochs = int(np.ceil(num_itr / len(self.learn.train_dl)))
        
        is_stop = False
        
        self.learn.model.train()
        
        for epoch in range(num_epochs):
        
            for xb, yb in self.learn.train_dl:

                batch_num += 1
                if batch_num > num_itr: 
                    is_stop = True
                    break

                yhatb = self.learn.model(xb.float())
                lossb = self.learn.loss(yhatb, yb.float())

                # compute the smoothed loss
                avg_loss = beta * avg_loss + (1-beta) * float(lossb)
                smoothed_loss = avg_loss / (1 - beta ** batch_num)
                
                # stop if the loss is exploding
                if smoothed_loss > 4 * best_loss: 
                    is_stop = True
                    break
                
                # record the best loss
                if smoothed_loss < best_loss: best_loss = smoothed_loss
                
                # store the values
                losses.append(smoothed_loss); lrs.append(lr)

                lossb.backward()
                self.learn.opt.step()
                self.learn.opt.zero_grad()

                lr *= mult
                for param_group in self.learn.opt.param_groups:
                    param_group['lr'] = lr
                    
            if is_stop: break
                
        self.plot_lrs_and_losses(
            lrs[skip_start:-skip_end], 
            losses[skip_start:-skip_end],
            init_value,
            final_value,
            show_suggestion
        )
        
    def set_lr(self, lr):
        for param_group in self.learn.opt.param_groups:
            param_group['lr'] = lr
        
    def train(self, num_epoch=10):
        
        self.cb_handler.on_train_begin()
    
        for epoch in tqdm_notebook(range(num_epoch)):

            self.cb_handler.on_epoch_begin()

            # ========== train ==========

            self.learn.model.train()
            for xb, yb in self.learn.train_dl:

                yhatb = self.learn.model(xb.float())
                lossb = self.learn.loss(yhatb, yb.float())

                lossb.backward()
                self.learn.opt.step()
                self.learn.opt.zero_grad()

            # ========== validation ==========

            self.learn.model.eval()
            for xb, yb in self.learn.valid_dl:
                
                self.cb_handler.on_batch_begin(yb=yb, bs=yb.size(0))  
                # yb: for computing accuracy (in AccuracyCallback)
                # bs: for weighting any metric
                
                yhatb = self.learn.model(xb.float())
                self.cb_handler.on_forward_end(yhatb=yhatb)  
                # yhatb: for computing accuracy (in AccuracyCallback)
                
                lossb = self.learn.loss(yhatb, yb.float())
                self.cb_handler.on_loss_end(lossb=lossb)  
                # lossb: for plotting loss  (in LossCallback)
                
                self.cb_handler.on_batch_end()
                
            self.cb_handler.on_epoch_end()
            
        self.cb_handler.on_train_end()
        
    def save_as_pth(self, path):
        torch.save(self.learn.model.state_dict(), path)