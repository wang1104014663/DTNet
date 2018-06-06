
import torch.nn as nn
import torch.nn.functional as F


class AttributeNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, input_size, hidden_size, output_size):
        super(AttributeNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)   #FC1 network
        self.fc2 = nn.Linear(hidden_size, output_size)  #FC2 network

    def forward(self, x):

        x = F.relu(self.fc1(x))      #activate
        x = F.relu(self.fc2(x))
        return x

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size, hidden_size,):
        super(RelationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  #FC3
        self.fc2 = nn.Linear(hidden_size, 1)           #FC4

    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class TripletNetwork(nn.Module):
    def __init__(self, embeddingnet_feature):
        super(TripletNetwork, self).__init__()
        #self.embeddingnet_att = embeddingnet_att
        self.embeddingnet_feature = embeddingnet_feature

    def forward(self, x, y, z):
        embedded_x = x
        embedded_y = self.embeddingnet_feature(y)
        embedded_z = self.embeddingnet_feature(z)
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b, embedded_x, embedded_y, embedded_z