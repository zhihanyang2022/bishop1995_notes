class Learner():
    
    def __init__(self, train_dl, valid_dl, model, loss, opt):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.model = model
        self.loss = loss
        self.opt = opt