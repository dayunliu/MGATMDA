import torch
import torch.nn as nn
import torch.nn.functional as F


class combiner(nn.Module):
    def __init__(self, embedding1, embedding2, embedding3,  embedding4, embedding_dim, droprate, cuda = 'cpu'):
        super(combiner, self).__init__()

        self.embedding1 = embedding1
        self.embedding2 = embedding2
        self.embedding3 = embedding3
        self.embedding4 = embedding4

        self.embed_dim = embedding_dim
        self.droprate = droprate
        self.device = cuda

        self.att1 = nn.Linear(self.embed_dim * 4, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, 4)
        self.softmax = nn.Softmax()

    def forward(self, nodes_u, nodes_i):
        embedding1 = self.embedding1(nodes_u, nodes_i)
        embedding2 = self.embedding2(nodes_u, nodes_i)
        embedding3 = self.embedding3(nodes_u, nodes_i)
        embedding4 = self.embedding4(nodes_u, nodes_i)

        x = torch.cat((embedding1, embedding2, embedding3, embedding4), dim = 1)
        x = F.relu(self.att1(x).to(self.device), inplace = True)
        x = F.dropout(x, training = self.training)
        x = self.att2(x).to(self.device)

        att_w = F.softmax(x, dim = 1)
        att_w1, att_w2, att_w3, att_w4= att_w.chunk(4, dim = 1)
        att_w1.repeat(self.embed_dim, 1)
        att_w2.repeat(self.embed_dim, 1)
        att_w3.repeat(self.embed_dim, 1)
        att_w4.repeat(self.embed_dim, 1)

        final_embed_matrix = torch.mul(embedding1, att_w1) + torch.mul(embedding2, att_w2)\
                             + torch.mul(embedding3, att_w3)\
                             + torch.mul(embedding4, att_w4)


        return final_embed_matrix