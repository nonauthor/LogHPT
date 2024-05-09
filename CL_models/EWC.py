import torch
from copy import deepcopy

from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
import torch.optim as optim
import torch.utils.data

from torchvision import datasets, transforms
import random


class OnlineEWC(object):
    def __init__(self, model, model_old, device, alpha=0.01, fisher=None):

        self.model = model
        self.model_old = model_old
        self.model_old_dict = self.model_old.state_dict()

        self.device = device

        self.fisher = {}
        if fisher is not None:  # initialize as old Fisher Matrix
            self.fisher_old = fisher
            for key in self.fisher_old:
                self.fisher_old[key].requires_grad = False
                self.fisher_old[key] = self.fisher_old[key].to(device)
                self.fisher[key] = torch.zeros_like(fisher[key], device=device)
        else:  # initialize a new Fisher Matrix
            self.fisher_old = None
            self.fisher = {n: torch.zeros_like(p, device=device, requires_grad=False)
                           for n, p in self.model.named_parameters() if p.requires_grad}

    # def update(self, dataloader, task):
    def update(self, dataloader):
        self.model.eval()
        for step, batch in enumerate(dataloader):
            self.model.zero_grad()
            # input = input.to(self.device)
            # target = target.to(self.device)
            #
            # # output = self.model(input, task)
            # output = self.model(input)
            #
            # loss = F.cross_entropy(output, target)  # Why they use entropy loss?
            # loss.backward()
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()



            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += p.grad.data.clone().pow(2) / len(dataloader)

    def get_fisher(self):
        return self.fisher  # return the new Fisher matrix

    def penalty(self):
        loss = 0
        if self.fisher_old is None:
            return 0.
        for n, p in self.model.named_parameters():
            loss += (self.fisher_old[n] * (p - self.model_old_dict[n]).pow(2)).sum()
        return loss

