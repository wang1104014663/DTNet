

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardTripletLoss(nn.Module):
    """Hard/Hardest Triplet Loss
    (pytorch implementation of https://omoindrot.github.io/triplet-loss)

    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    """
    def __init__(self, margin= 1 ):
        """
        Args:
            margin: margin for triplet loss
            hardest: If true, loss is considered only hardest triplets.
            squared: If true, output is the pairwise squared euclidean distance matrix.
                If false, output is the pairwise euclidean distance matrix.
        """
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        
        
    def forward(self, attributes, embeddings, labels):
        """
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        
        relations = _pairwise_distance(attributes, embeddings ,labels)
        #print(relations.size())
        
        # Get the hardest positive pairs
        mask_pos=_get_anchor_positive_triplet_mask(relations, labels).float()
        #print(mask_pos)
        valid_positive_dist = relations * mask_pos
        hardest_positive_dist, _ = torch.max(valid_positive_dist, 0 ,keepdim=True)
        


        # Get the hardest negative pairs
        mask_neg=_get_anchor_negative_triplet_mask(relations, labels).float()
        #print(mask_neg)
        max_anchor_negative_dist, _ = torch.max(relations, 0 , keepdim=True)
        anchor_negative_dist = relations + max_anchor_negative_dist * (1.0 - mask_neg)
        

        hardest_negative_dist, _ = torch.min(anchor_negative_dist, 0 , keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        #print(hardest_negative_dist)
        triplet_loss_all = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        num_hard_triplets = torch.sum(torch.gt(triplet_loss_all, 1e-16).float())
        triplet_loss = torch.sum(triplet_loss_all) / (num_hard_triplets + 1e-16)
        #triplet_loss = torch.mean(triplet_loss)
        #triplet_loss.requires_grad=True
        
        #if triplet_loss != torch.mean(triplet_loss_all):
            #print('not equal \n','-'*100)
            #print('triplet_loss: ',triplet_loss)
            #print('triplet_mean',torch.mean(triplet_loss_all))
            #print('-'*100)
        return triplet_loss
def _pairwise_distance(bat_attributes, bat_features, bat_lables):
    # Compute the 2D matrix of distances between all the embeddings.
    distances = F.pairwise_distance(bat_features,bat_attributes,2).view(bat_lables.size()[0],-1)
    return distances
def _get_anchor_positive_triplet_mask(relations,labels):
    # Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mask_pos = torch.zeros_like(relations).to(device).byte()
    for i in range(relations.size()[0]):
        mask_pos[i][labels[i]]=1

    return mask_pos
def _get_anchor_negative_triplet_mask(relations,labels):
    # Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Check if labels[i] != labels[k]
    mask_neg = torch.ones_like(relations).to(device).byte()
    #print('-'*100)
    for i in range(relations.size()[0]):
        mask_neg[i][labels[i]]=0

    return mask_neg

