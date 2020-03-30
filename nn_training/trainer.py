from tqdm import tqdm_notebook

class Trainer:
    
    def __init__(self, learn, cb_handler):
        self.learn = learn
        self.cb_handler = cb_handler
        
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