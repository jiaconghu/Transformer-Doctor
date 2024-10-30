import torch
from metrics import Metric

from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score
from torchmetrics import Accuracy, Recall, Specificity
from torchmetrics.classification.stat_scores import MulticlassStatScores, Tensor


def my_safe_divide(num: Tensor, denom: Tensor) -> Tensor:
    denom[denom == 0.0] = 1
    num = num if num.is_floating_point() else num.float()
    denom = denom if denom.is_floating_point() else denom.float()
    return num / denom


class MulticlassFalsePositiveRate(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.false_positives = 1 - Specificity(task='multiclass', average='macro', num_classes=num_classes)

    def update(self, outputs, targets):
        self.false_positives.update(outputs, targets)

    def compute(self):
        FPR = self.false_positives.compute()
        return FPR

    def to(self, device):
        self.false_positives.to(device)
        return self
    

class MulticlassFNR(MulticlassStatScores):
    def compute(self) -> Tensor:
        """Computes accuracy based on inputs passed in to ``update`` previously."""
        tp, fp, tn, fn = self._final_state()
        return my_safe_divide(fn , fn + tp)


class MulticlassFalseNegativeRate(Metric):
    def __init__(self, average, num_classes):
        super().__init__()
        self.avg = average
        # self.false_negatives = 1 - Recall(task='multiclass', average='micro', num_classes=num_classes)
        self.false_negatives = MulticlassFNR(num_classes=num_classes, average=average)

    def update(self, outputs, targets):
        self.false_negatives.update(outputs, targets)

    def compute(self):
        FNR = self.false_negatives.compute()
        if self.avg == 'macro':
            tmp = 0.0
            for i in FNR:
                tmp += i
            tmp /= 1000
            return tmp
        else:
            return FNR

    def to(self, device):
        self.false_negatives.to(device)
        return self


class MulticlassFOR(MulticlassStatScores):
    def compute(self) -> Tensor:
        """Computes accuracy based on inputs passed in to ``update`` previously."""
        tp, fp, tn, fn = self._final_state()
        return my_safe_divide(fn, fn + tn)


class MulticlassFalseOmissionRate(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.false_omission = MulticlassFOR(num_classes=num_classes, average='macro')

    def update(self, outputs, targets):
        self.false_omission.update(outputs, targets)

    def compute(self):
        FOR = self.false_omission.compute()
        return FOR

    def to(self, device):
        self.false_omission.to(device)
        return self
    

class MulticlassFDR(MulticlassStatScores):
    def compute(self) -> Tensor:
        """Computes accuracy based on inputs passed in to ``update`` previously."""
        tp, fp, tn, fn = self._final_state()
        return my_safe_divide(fp, fp + tp)


class MulticlassFalseDiscoveryRate(Metric):
    def __init__(self, average, num_classes):
        super().__init__()
        self.avg = average
        self.false_discovery = MulticlassFDR(num_classes=num_classes, average=average)

    def update(self, outputs, targets):
        self.false_discovery.update(outputs, targets)

    def compute(self):
        FDR = self.false_discovery.compute()
        if self.avg == 'macro':
            tmp = 0.0
            for i in FDR:
                tmp += i
            tmp /= 1000
            return tmp
        else:
            return FDR

    def to(self, device):
        self.false_discovery.to(device)
        return self


# class MulticlassFalsePositiveRate(Metric):
#     def __init__(self):
#         super().__init__()
#         self.false_positives = torch.tensor(0)
#         self.true_negatives = torch.tensor(0)

#     def update(self, outputs, targets):
#         targets = torch.reshape(targets, outputs.shape)
#         false_positives = torch.logical_and(outputs == 1, targets == 0).sum()
#         true_negatives = torch.logical_and(outputs == 0, targets == 0).sum()
#         self.false_positives += false_positives
#         self.true_negatives += true_negatives

#     def compute(self):
#         return self.false_positives / (self.false_positives + self.true_negatives)


# class MulticlassFalseNegativeRate(Metric):
#     def __init__(self):
#         super().__init__()
#         self.false_negatives = torch.tensor(0)
#         self.true_positives = torch.tensor(0)

#     def update(self, outputs, targets):
#         targets = torch.reshape(targets, outputs.shape)
#         false_negatives = torch.logical_and(outputs == 0, targets == 1).sum()
#         true_positives = torch.logical_and(outputs == 1, targets == 1).sum()
#         self.false_negatives += false_negatives
#         self.true_positives += true_positives

#     def compute(self):
#         return self.false_negatives / (self.false_negatives + self.true_positives)


class MulticlassBalancedAccuracy(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.sensitivity = Recall(task='multiclass', average='macro', num_classes=num_classes)
        self.specificity = Specificity(task='multiclass', average='macro', num_classes=num_classes)

    def update(self, outputs, targets):
        self.sensitivity.update(outputs, targets)
        self.specificity.update(outputs, targets)

    def compute(self):
        ba = (self.sensitivity.compute() + self.specificity.compute()) / 2
        return ba

    def to(self, device):
        self.sensitivity.to(device)
        self.specificity.to(device)
        return self


class MulticlassOptimizedPrecision(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.accuracy = Accuracy(task='multiclass', average='macro', num_classes=num_classes)
        self.sensitivity = Recall(task='multiclass', average='macro', num_classes=num_classes)
        self.specificity = Specificity(task='multiclass', average='macro', num_classes=num_classes)

    def update(self, outputs, targets):
        self.accuracy.update(outputs, targets)
        self.sensitivity.update(outputs, targets)
        self.specificity.update(outputs, targets)

    def compute(self):
        sensitivity = self.sensitivity.compute()
        specificity = self.specificity.compute()
        op = self.accuracy.compute() - torch.abs(sensitivity - specificity) / (sensitivity + specificity)
        return op

    def to(self, device):
        self.accuracy.to(device)
        self.sensitivity.to(device)
        self.specificity.to(device)
        return self
