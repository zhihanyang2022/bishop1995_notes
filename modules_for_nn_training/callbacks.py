class Callback(object):
    """Base class for all utility callbacks and CallbackHandler."""
    
    def on_train_begin(self): pass
    def on_epoch_begin(self): pass
     
    # =================================
    
    def on_batch_begin(self): pass  # add xb and yb to state_dict
    def on_loss_end(self): pass
    def on_backward_begin(self): pass  # additional loss terms might be available (e.g., tangent prop)
    
    # =================================
    # callback functions for validation
    # =================================
    
    def eval_on_batch_begin(self): pass  # add xb and yb to state_dict
    def eval_on_forward_end(self): pass  # yhatb is available for computing accuracy
    def eval_on_loss_end(self): pass  # loss is available
    def eval_on_batch_end(self): pass
    
    # =================================
    
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
        self.state_dict['xb'] = kwargs['xb']  # e.g., tangent prop
        self.state_dict['yb'] = kwargs['yb']  # for computing accuracy
        self.state_dict['bs'] = kwargs['bs']
        
        for cb in self.cbs:
            cb.on_batch_begin()
            
    def on_loss_end(self, **kwargs):
        
        self.state_dict['lossb'] = kwargs['lossb']  # for computing loss
        
        for cb in self.cbs:
            cb.eval_on_loss_end()
            
    def on_backward_begin(self, **kwargs):
        
        #  model parameters might be required to calculate additional loss terms (e.g., if related to Jacobian / Hessian)
        self.state_dict['model'] = kwargs['model'] 
        
        for cb in self.cbs:
            cb.on_backward_begin()
            
    def eval_on_batch_begin(self, **kwargs):
        
        self.state_dict['batch_index'] += 1
        self.state_dict['xb'] = kwargs['xb']  # e.g., tangent prop
        self.state_dict['yb'] = kwargs['yb']  # for computing accuracy
        self.state_dict['bs'] = kwargs['bs']  # for computing any metric
        
        for cb in self.cbs:
            cb.eval_on_batch_begin()
            
    def eval_on_forward_end(self, **kwargs):
        
        self.state_dict['yhatb'] = kwargs['yhatb']  # for computing accuracy
        
        for cb in self.cbs:
            cb.eval_on_forward_end()
            
    def eval_on_loss_end(self, **kwargs): 
        
        self.state_dict['lossb'] = kwargs['lossb']  # for computing loss
        
        for cb in self.cbs:
            cb.eval_on_loss_end()
            
    def eval_on_batch_end(self): 
        for cb in self.cbs:
            cb.eval_on_batch_end()
            
    def on_epoch_end(self): 
        for cb in self.cbs:
            cb.on_epoch_end()
            
    def on_train_end(self): 
        for cb in self.cbs:
            cb.on_train_end()