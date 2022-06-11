from metric_base import Metric, Activation, f_score, accuracy, recall, precision

class Fscore(Metric):
    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return f_score(y_pr, y_gt, eps=self.eps, beta=self.beta, threshold=self.threshold)


class Accuracy(Metric):
    def __init__(self, threshold=0.5, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return accuracy(y_pr, y_gt, threshold=self.threshold)


class Recall(Metric):
    def __init__(self, eps=1e-7, threshold=0.5, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return recall(y_pr, y_gt, eps=self.eps, threshold=self.threshold)


class Precision(Metric):
    def __init__(self, eps=1e-7, threshold=0.5, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return precision(y_pr, y_gt, eps=self.eps, threshold=self.threshold)