import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import jaccard_score

class ContrastiveLoss(nn.Module):
    def __init__(self, device, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.device = device

    def forward(self, outputs: tuple[torch.tensor, torch.tensor], labels: torch.tensor) -> torch.tensor:
        output1, output2 = outputs
        output1 = output1.to(self.device)
        output2 = output2.to(self.device)
        labels = labels.to(self.device)
        distance = F.pairwise_distance(output1, output2)
        # print('Distance: ' + str(euclidean_distance))
        # print('Label: ' + str(label))
        
        # Assuming label is 1 for similar pairs and 0 for dissimilar pairs
        loss = (labels) * torch.pow(distance, 2) + (1-labels) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        # loss = torch.mean(loss)
        # loss = 0.5 * (label * output**2 + (1 - label) * F.relu(self.margin - output).pow(2))
        return loss.mean().to(self.device)

class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, device, scale: float = 1.0):
        super(MultipleNegativesRankingLoss, self).__init__()
        self.device = device
        self.scale = scale
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs: tuple[torch.tensor, torch.tensor], labels: torch.tensor) -> torch.tensor:
        output1, output2 = outputs
        output1 = output1.to(self.device)
        output2 = output2.to(self.device)
        labels = labels.to(self.device)
        scores = torch.cosine_similarity(output1, output2) * self.scale
        scores = scores.to(self.device)
        # Example a[i] should match with b[i]
        range_labels = torch.arange(0, scores.size(0), device=scores.device)
        return self.cross_entropy_loss(scores, labels.to(float))


class TripletLoss(nn.Module):
    def __init__(self, device, margin=1.0):
        super(TripletLoss, self).__init__()
        self.device = device
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)
    
    def forward(self, anchor_emb: torch.tensor, positive_emb: torch.tensor, negative_emb: torch.tensor) -> torch.tensor:
        return self.triplet_loss(anchor_emb, positive_emb, negative_emb).to(self.device)

class NTXentLoss(nn.Module):
    def __init__(self, device, batch_size, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self._get_mask(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.device = device

    def _get_mask(self, batch_size):
        # Create a mask to exclude diagonal elements
        mask = torch.ones((batch_size * 2, batch_size * 2), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, i + batch_size] = 0
            mask[i + batch_size, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        Args:
            z_i: Projection of first batch of samples (batch_size x dim)
            z_j: Projection of second batch of samples (batch_size x dim)
        """
        print(z_i)
        print(z_j)
        # Concatenate the projections from both augmented views
        z = torch.cat((z_i, z_j), dim=0)  # (2*batch_size, dim)

        # Compute similarity matrix
        sim = torch.mm(z, z.t())  # (2*batch_size, 2*batch_size)
        sim = sim / self.temperature  # Scale with temperature

        # Mask out the diagonal elements
        sim = sim[self.mask].view(2 * self.batch_size, -1)  # Remove diagonals

        # Create labels (0 to batch_size for z_i matches with batch_size to 2*batch_size for z_j)
        positive_samples = torch.cat([torch.arange(self.batch_size) + self.batch_size, torch.arange(self.batch_size)])
        positive_samples = positive_samples.to(sim.device)

        # Compute loss
        loss = self.criterion(sim, positive_samples)
        loss /= (2 * self.batch_size)  # Normalize by the number of samples

        return loss.to(self.device)

class CosineSimilarityLoss(nn.Module):
    def __init__(self, device, loss_fct: nn.Module = nn.BCEWithLogitsLoss(), cos_score_transformation: nn.Module = nn.Sigmoid(),) -> None:
        super(CosineSimilarityLoss, self).__init__()
        self.device = device
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation

    def forward(self, outputs: tuple[torch.tensor, torch.tensor], labels: torch.tensor) -> torch.tensor:
        output = self.cos_score_transformation(torch.cosine_similarity(outputs[0], outputs[1]))
        return self.loss_fct(output, labels.float().view(-1)) 


def normal_nll(mu, sigma, x):
    nll =  torch.sum( torch.log(sigma) ) + torch.mul( torch.div( x.size(0), 2.0 ), torch.log( torch.mul(2, np.pi) ) ) + torch.sum( torch.div( torch.square( torch.sub(x, mu) ), torch.mul( 2, torch.square(sigma) ) ) )
    
    return nll

def training_step(batch: dict, batch_idx: int, len_data: int, model: nn.modules, criterion: nn.modules, optimizer: torch.optim, configs: dict, device: torch.device) -> dict:
    # A1 = batch['A1']
    # A2 = batch['A2']
    # B1 = batch['B1']
    # B2 = batch['B2']
    inputs = batch['inputs']
    labels = batch['labels']
    # print('Label: ' + str(labels))
    
    # outputs = model(A1, A2, B1, B2)
    outputs = model(inputs)
    labels = labels.to(device).to(torch.float32)
    # print(outputs.shape, label.shape)
    # print(outputs.dtype, label.dtype)
    loss = criterion(outputs, labels) / configs.accumulation_steps

    loss.backward()
    
    # Gradient accumulation step for efficiency
    if ((batch_idx + 1) % configs.accumulation_steps == 0) or (batch_idx + 1 == len_data):
        # print('inside optimizer')
        optimizer.step()
        optimizer.zero_grad()

    log = {'loss': loss.cpu().detach(), 'output1': outputs[0].cpu().detach()}
    return log
