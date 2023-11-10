import torch
import torch.nn as nn
def corr_coeff(X):
    X_m = (X - X.mean(dim=(0, 1), keepdim=True))
    X_m = torch.nn.functional.normalize(X_m)
    XX = 1 - torch.bmm(torch.permute(X_m, (2, 0, 1)), torch.permute(X_m, (2, 1, 0)))
    return XX

def normalize(X):
    norm = torch.norm(X, p=2, dim=1)
    X = X / norm.unsqueeze(1)
    return X


def similarity_loss(X, Y):
    XX=corr_coeff(X)
    #XX= torch.clamp(XX, 0.0, np.inf)
    YY=corr_coeff(Y)
    # get upper diagonal
    n1 = XX.shape[1]
    pairs = torch.combinations(torch.arange(n1), with_replacement=False)
    XX_vec=XX[:,pairs[:, 0], pairs[:, 1]]
    YY_vec=YY[:,pairs[:,0],pairs[:,1]]
    # compute cosine similarity
    XX_vec=normalize(XX_vec)
    YY_vec=normalize(YY_vec)
    #
    similarites=1-torch.diag(XX_vec @ YY_vec.T)
    #XY_loss=torch.sum(similarites)+torch.var(similarites)
    return similarites


class CustomLayer(nn.Module):
    def __init__(self,n_channels=7,n_features=650,n_hidden=256):
        super(CustomLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(n_channels, n_features, n_hidden))
        self.bias = nn.Parameter(torch.randn(n_channels, n_hidden))
    def forward(self, input_data):
        #output=torch.einsum('bic,cio->bco', input_data, self.weight)+self.bias
        output = torch.einsum('bic,cio->bco', input_data, self.weight) + self.bias
        # reshape output to be batch x features x channels
        #output = output.permute(0, 2, 1)
        return output
