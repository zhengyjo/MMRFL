from Utils.AvgMeter import AvgMeter
from tqdm import tqdm
from Utils.mr2mr import *
import torch
def train_epoch(model, train_loader, optimizer, lr_scheduler, step, accuracies_req):
    loss_meter = AvgMeter(len(accuracies_req))
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    for batch in tqdm_object:
        for key, value in batch.items():
            if key != 'smiles_input':
                batch[key] = value.to(model.device)

        # Assuming 'model.device' is the target device (e.g., 'cuda' for GPU or 'cpu' for CPU)
        loss, nodeLoss, graphLoss, graphLogits = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        # do not allow gradient in calculating accuracy.
        graph_acc = mr2mr_match(graphLogits.detach(), accuracies_req)

        loss_meter.update(loss.item(), nodeLoss.item(), graphLoss.item(), graph_acc)
        loss_meter.get_lr(optimizer)

        tqdm_object.set_postfix(
            train_loss=loss_meter.avg,
            nodeLoss=loss_meter.nodeLoss_avg,
            graphLoss_loss=loss_meter.graphLoss_avg,
            graph_acc=loss_meter.graph_acc_avg,
            lr=loss_meter.lr
        )
        #loss_meter.print_epoch_results() 

    return loss_meter
