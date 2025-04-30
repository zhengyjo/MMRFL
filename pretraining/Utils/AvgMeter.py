import torch
class AvgMeter:
    def __init__(self, accuracies_req_num=1):
        self.name = "Metric"
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.lr = 0.0
        self.graph_acc_avg = torch.zeros(accuracies_req_num)
        self.graph_acc_sum = torch.zeros(accuracies_req_num)

        self.nodeLoss_avg = 0.0
        self.nodeLoss_sum = 0.0
        self.graphLoss_avg = 0.0
        self.graphLoss_sum = 0.0

    def update(self, loss, nodeLoss, graphLoss, graph_acc):
        self.count += 1

        self.sum += loss
        self.avg = self.sum / self.count

        self.graphLoss_sum += graphLoss
        self.graphLoss_avg = self.graphLoss_sum/self.count
 
        self.nodeLoss_sum += nodeLoss
        self.nodeLoss_avg = self.nodeLoss_sum/self.count

        self.graph_acc_sum += graph_acc

        self.graph_acc_avg = self.graph_acc_sum/self.count


    def get_lr(self, optimizer):
        self.lr = optimizer.param_groups[0]['lr']
        return self.lr

