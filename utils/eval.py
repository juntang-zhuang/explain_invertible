from __future__ import print_function, absolute_import
import torch
__all__ = ['accuracy','attribute_accuracy']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def attribute_accuracy(output,target):
    N,C,_ = output.size()
    pred_ind = torch.argmax(output,dim=-1)
    pred_ind2 = 1 - pred_ind

    pred = torch.stack((pred_ind2,pred_ind),-1).float()

    accuracy = torch.sum(target * pred) / float(N*C)

    return accuracy.item()