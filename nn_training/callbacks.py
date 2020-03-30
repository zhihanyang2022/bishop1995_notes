class Callback(object):
    """Base class for all utility callbacks and CallbackHandler."""
    
    def on_train_begin(self): pass
    def on_epoch_begin(self): pass
    def on_batch_begin(self): pass
    
    def on_forward_end(self): pass  # yhatb is available for computing accuracy
    def on_loss_end(self): pass  # loss is available
    
    def on_batch_end(self): pass
    def on_epoch_end(self): pass
    def on_train_end(self): pass
    
class CallbackHandler(Callback):
    
    def __init__(self, cbs): 
        self.cbs = cbs
        self.state_dict = {}
        for c in self.cbs:  # each child cb can see self.state_dict and its updated versions
            c.state_dict = self.state_dict
    
    def on_train_begin(self): 
        
        self.state_dict['epoch'] = -1
        
        for cb in self.cbs:
            cb.on_train_begin()
            
    def on_epoch_begin(self): 
        
        self.state_dict['epoch'] += 1
        self.state_dict['batch_index'] = -1
        
        for cb in self.cbs:
            cb.on_epoch_begin()
            
    def on_batch_begin(self, **kwargs):
        
        self.state_dict['batch_index'] += 1
        
        self.state_dict['yb'] = kwargs['yb']  # for computing accuracy
        self.state_dict['bs'] = kwargs['bs']  # for computing any metric
        
        for cb in self.cbs:
            cb.on_batch_begin()
            
    def on_forward_end(self, **kwargs):
        
        self.state_dict['yhatb'] = kwargs['yhatb']  # for computing accuracy
        
        for cb in self.cbs:
            cb.on_forward_end()
            
    def on_loss_end(self, **kwargs): 
        
        self.state_dict['lossb'] = kwargs['lossb']  # for computing loss
        
        for cb in self.cbs:
            cb.on_loss_end()
            
    def on_batch_end(self): 
        for cb in self.cbs:
            cb.on_batch_end()
            
    def on_epoch_end(self): 
        for cb in self.cbs:
            cb.on_epoch_end()
            
    def on_train_end(self): 
        for cb in self.cbs:
            cb.on_train_end()