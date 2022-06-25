from modelutil import *

class Fscore(Metrics):
    def __init__(self, **kwargs):
        super(Fscore, self).__init__(**kwargs)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return f_score(y_pr, y_gt, eps=self.eps, beta=self.beta, threshold=self.threshold)

class Accuracy(Metrics):
    def __init__(self, **kwargs):
        super(Accuracy, self).__init__(**kwargs)
        
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return accuracy(y_pr, y_gt, threshold=self.threshold)


class Recall(Metrics):
    def __init__(self, **kwargs):
        super(Recall, self).__init__(**kwargs)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return recall(y_pr, y_gt, eps=self.eps, threshold=self.threshold)


class Precision(Metrics):
    def __init__(self, **kwargs):
        super(Precision, self).__init__(**kwargs)
        
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return precision(y_pr, y_gt, eps=self.eps, threshold=self.threshold)
    


