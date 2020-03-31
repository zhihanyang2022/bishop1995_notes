import math
import numpy as np
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt

class Trainer:
    
    def __init__(self, learn, cb_handler):
        self.learn = learn
        self.cb_handler = cb_handler
        
    def find_lr(self, init_value = 1e-7, final_value=10., beta = 0.98):
        
        num = 200
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        
        for param_group in self.learn.opt.param_groups:
            param_group['lr'] = lr
        
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        lrs = []
        
        num_epochs = int(np.ceil(num / len(self.learn.train_dl)))
        
        self.learn.model.train()
        
        for epoch in range(num_epochs):
        
            for xb, yb in self.learn.train_dl:

                batch_num += 1
                if batch_num > num:
                    return lrs, losses

                yhatb = self.learn.model(xb.float())
                lossb = self.learn.loss(yhatb, yb.float())

                #Compute the smoothed loss
                avg_loss = beta * avg_loss + (1-beta) * float(lossb)
                smoothed_loss = avg_loss / (1 - beta**batch_num)
                #Stop if the loss is exploding
                if batch_num > 1 and smoothed_loss > 4 * best_loss:
                    return lrs, losses
                #Record the best loss
                if smoothed_loss < best_loss or batch_num==1:
                    best_loss = smoothed_loss
                #Store the values
                losses.append(smoothed_loss)
                lrs.append(lr)

                lossb.backward()
                self.learn.opt.step()
                self.learn.opt.zero_grad()

                lr *= mult
                for param_group in self.learn.opt.param_groups:
                    param_group['lr'] = lr
            
        return lrs, losses
        
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