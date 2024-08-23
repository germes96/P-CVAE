
"""Library implementing prototype based variational auto-encoder

Authors
 * St Germes Bengono Obiang 2023
 * Norbert Tsopze 2023
"""

#import core library
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn import Linear
import torch
#select execution device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

## Definie list of function for compute data similarity
def list_of_norms(X):
    '''
    X is a list of vectors X = [x_1, ..., x_n], we return
        [d(x_1, x_1), d(x_2, x_2), ... , d(x_n, x_n)], where the distance
    function is the squared euclidean distance.
    '''
    return torch.sum(torch.pow(X, 2), axis=1)

def list_of_distance(X: Tensor, Y: Tensor) -> Tensor:
    """
    Compute the matrix of pairwise squared Euclidean distances between two lists of vectors.

    Given a list of vectors, X = [x_1, ..., x_n], and another list of vectors,
    Y = [y_1, ... , y_m], we return a list of vectors
            [[d(x_1, y_1), d(x_1, y_2), ... , d(x_1, y_m)],
             ...
             [d(x_n, y_1), d(x_n, y_2), ... , d(x_n, y_m)]],
    where the distance metric used is the sqared euclidean distance.
    The computation is achieved through a clever use of broadcasting.
    
    Parameters
    ----------
    X : Tensor
        The first list of vectors (n, d)
    Y : Tensor
        The second list of vectors (m, d)
    
    Returns
    -------
    Tensor
        The matrix of pairwise distances (n, m)
    """
    # Compute the norms of the vectors in X and Y
    XX = list_of_norms(X).unsqueeze(1)  # (n, 1)
    YY = list_of_norms(Y).unsqueeze(0)  # (1, m)
    
    # Compute the squared Euclidean distances using broadcasting
    output = XX + YY - 2 * torch.matmul(X, Y.T)
    return output


#Variatinal encoder class
class VariationalEncoder(nn.Module):
    """
    The variational encoder is implemented as described in Li et al 2019. It is composed of 3 convolutional layers and two dence layers. 
    
    Arguments
    ---------
    latens_dims : int 
        The size of the latent dim. this size will be use as prototype dim
    in_channels : int
        Number of input channel. by default x-vector is on one dimention
    """

    """
    The variational encoder is implemented as described in Li et al 2019.
    It is composed of 3 convolutional layers and two dence layers.
    
    Parameters
    ----------
    latent_dims : int
        The size of the latent dim. this size will be use as prototype dim
    input_shape : tuple
        The shape of the input data
    in_channels : int
        Number of input channel. by default x-vector is on one dimention
    """
    def __init__(self, latent_dims, input_shape,  in_channels=1):
        """
        Initialize the VariationalEncoder module
        
        Parameters
        ----------
        latent_dims : int
            The size of the latent dim. this size will be use as prototype dim
        input_shape : tuple
            The shape of the input data
        in_channels : int
            Number of input channel. by default x-vector is on one dimention
        """
        super(VariationalEncoder, self).__init__()

        self.feature_extractor = torch.torch.nn.Sequential(
            # Conv1d layer with 8 filters, kernel size 3 and padding 1
            nn.Conv1d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1, bias=False),
            # ReLU activation
            nn.ReLU(),
            # Conv1d layer with 16 filters, kernel size 3 and padding 2
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=2, bias=False),
            # Batch normalization
            nn.BatchNorm1d(16),
            # ReLU activation
            nn.ReLU(),
            # Conv1d layer with 32 filters, kernel size 3, stride 2 and padding 0
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=0),
            # ReLU activation
            nn.ReLU(),
            # Flatten layer
            nn.Flatten(start_dim=1),
        )
        self.linear_shape = list(self.feature_extractor(torch.rand(input_shape)).shape)[1]

        # Dense layer with 128 units
        self.linear1 = torch.nn.Linear(self.linear_shape, 128)
        # Dense layer with latent_dim units
        self.linear2 = nn.Linear(128, latent_dims)
        # Dense layer with latent_dim units
        self.linear3 = nn.Linear(128, latent_dims)

        # Normal distribution for the prior
        self.N = torch.distributions.Normal(0, 1)
        # Hack to get sampling on the GPU
        self.N.loc = self.N.loc
        self.N.scale = self.N.scale
        # KL loss
        self.kl = 0
    
    def reparameterize(self, mu: Tensor, sigma: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).

        The reparameterization trick is a method to sample from a distribution
        by sampling from a standard normal distribution and then applying a
        transformation to the sample to obtain a sample from the desired
        distribution.

        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param sigma: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        # Compute the standard deviation of the latent gaussian
        std = torch.exp(0.5 * sigma)
        # Sample from a standard normal distribution
        eps = torch.randn_like(std)
        # Apply the transformation to the sample to obtain a sample from the
        # desired distribution
        return eps * std + mu

    def forward(self, x: Tensor, label: Tensor) -> Tensor:
        """
        The forward pass of the encoder.

        This function takes an input tensor and a label and applies the encoding
        transformation to the input tensor. The reparameterization trick is used
        to sample from the latent distribution.

        :param x: The input tensor [B x D]
        :param label: The label of the input tensor [B]
        :return: The latent tensor [B x D]
        """
        x = x.to(device)
        # Extract the features of the input
        x = self.feature_extractor(x)
        # Apply the first linear layer
        x = F.relu(self.linear1(x))
        # Compute the mean and the log of the variance of the latent gaussian
        mu =  self.linear2(x)
        sigma = self.linear3(x)
        # Sample from the latent gaussian
        z = self.reparameterize(mu, sigma)
        # Compute the KL loss
        self.kl = -0.5 *torch.mean(torch.sum(1 + sigma - mu ** 2 - sigma.exp(), dim = 1), dim = 0)
        # Return the latent tensor
        return z
 
## Classic Decoder
class Decoder(nn.Module):
    """
    The decoder's role is to reconstruct the data in the latent space of the variational encoder.
    
    Arguments
    ---------
    latens_dims : int
        The size of the latent dim. this size will be use as prototype dim
    out_channels : int
        Number of output channels. corresponds to the number of input channels
    """  
    """
    The decoder's role is to reconstruct the data in the latent space of the variational encoder.

    :param latent_dims: The size of the latent dim. this size will be use as prototype dim
    :param out_channels: Number of output channels. corresponds to the number of input channels
    """
    def __init__(self, latent_dims, out_channels, output_shape):
        """
        Initialize the decoder.

        :param latent_dims: The size of the latent dim. this size will be use as prototype dim
        :param out_channels: Number of output channels. corresponds to the number of input channels
        :param output_shape: The output shape of the decoder
        """
        super(Decoder, self).__init__()

        # The first layer is a linear layer followed by a ReLU activation
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.ReLU(True)
        )

        # The second layer is a linear layer followed by a ReLU activation
        self.linear1 = nn.Sequential(
            nn.Linear(128, output_shape),
            nn.ReLU(True)
        )

        # The third layer is a convolutional transpose layer
        # The output shape is (32, output_shape/32)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, int(output_shape/32)))

        # The fourth layer is a convolutional transpose layer
        self.d_conv1 = nn.ConvTranspose1d(32, 16, 3, stride=2, output_padding=1)

        # The fifth layer is a convolutional transpose layer
        self.d_conv2 = nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=3, padding=2, bias=False)

        # The sixth layer is a batch normalization layer
        self.batch2 = nn.BatchNorm1d(16)

        # The seventh layer is a convolutional transpose layer
        self.d_conv3 = nn.ConvTranspose1d(in_channels= 8, out_channels=out_channels, kernel_size=3, padding=1 ,output_padding=0)

    def forward(self, x):
        """
        The forward pass of the decoder.

        :param x: The input to the decoder
        :return: The output of the decoder
        """
        # The first layer is a linear layer followed by a ReLU activation
        x = self.decoder_lin(x)

        # The second layer is a linear layer followed by a ReLU activation
        x = self.linear1(x)

        # The third layer is a convolutional transpose layer
        # The output shape is (32, output_shape/32)
        x = self.unflatten(x)

        # The fourth layer is a convolutional transpose layer
        x = F.relu(self.d_conv1(x))

        # The fifth layer is a convolutional transpose layer
        x = F.relu(self.d_conv2(self.batch2(x)))

        # The sixth layer is a convolutional transpose layer
        x = F.relu(self.d_conv3(x))

        # The seventh layer is a sigmoid activation
        x = torch.sigmoid(x)

        return x

#PrototypeLayer   
class PrototypeLayer(nn.Module):
    """
    the prototyping layer consists of a learnable matrix. each column of this matrix is a prototype. 

    Arguments
    ---------
    prototype_dims: int 
        The dimention of each protype. this corresponds to the size of the variational encoder's latent space.
    n_prototypes : int
        The nomber of prototype. greater than or equal to the number of classes to be prototyped
    """
    def __init__(self, n_prototypes, latent_dims):
        super(PrototypeLayer, self).__init__()
        self.n_prototypes = n_prototypes
        self.prototypes = torch.nn.Parameter(torch.rand([n_prototypes, latent_dims]), requires_grad=True)
        self.R_1 = 0 #Feature distances regulation
        self.R_2 = 0 #Prototypes distances regulation
        self.R_3 = 0 #Prototypes to prototypes distances
        self.laten_similarity = 0 # Prototypes distances
    def computeRegulation(self, distance_list):
        """
        This function compute the regulation value for the prototypes.

        The regulation value is the mean of the minimum distance of each prototype to the closest feature.
        """
        min = torch.min(distance_list, dim=1)
        return torch.mean(min.values)
    
    def forward(self, x):
        """
        The forward pass in the prototyping layer is just a computation of the regulation values.
        The regulation values are the mean of the minimum distance of each prototype to the closest feature.
        """
        #Compute distance
        '''
        For each prototype we compute the distance to every feature
        '''
        distance_to_features = list_of_distance(self.prototypes, x)
        # print(x.shape, self.prototypes.shape)
        self.R_1 = torch.mean(torch.min(distance_to_features, dim=1).values)
        '''
        For each feature we compute the distance to every prototype
        '''
        distance_to_prototypes = list_of_distance(x, self.prototypes)
        self.R_2 = torch.mean(torch.min(distance_to_prototypes, dim=1).values)
        proto_to_proto = list_of_distance(self.prototypes, self.prototypes)
        #remove all zero value
        proto_to_proto_non_zero = proto_to_proto.clone()
        proto_to_proto_non_zero[proto_to_proto==0] = 1e+6
        self.R_3 = torch.mean(torch.min(proto_to_proto_non_zero, dim=1).values)

        self.laten_similarity = distance_to_prototypes
        if not torch.isfinite(distance_to_prototypes).all():
            print(distance_to_prototypes)
            raise InterruptedError("invilid data")
    
        y = self.prototypes * torch.sum(x) #for fun
        return x

# Variational auto encoder
class VariationalAutoencoder(nn.Module):
    """This module realizes the principle of variational autoencoding with prototypes. 
        This module implements the principle of variational autoencoding with prototypes. It therefore couples the autoencoder layer, the decoder and the prototyping layer.
    """
    def __init__(self, latent_dims, n_prototypes, in_channels, input_shape):
        super(VariationalAutoencoder, self).__init__()
        #ENCODER LAYER
        self.encoder = VariationalEncoder(latent_dims=latent_dims, in_channels=in_channels, input_shape = input_shape)
        #PROTOTYPE LAYER
        self.proto = PrototypeLayer(latent_dims=latent_dims, n_prototypes=n_prototypes)
        #DECODER LAYER
        self.decoder = Decoder(latent_dims=latent_dims, out_channels=in_channels, output_shape =  self.encoder.linear_shape)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x, label = None)
        #send data to prototype layer to compute distance
        _ = self.proto(z)
        return self.decoder(z), self.proto.prototypes , self.proto.laten_similarity


# Variational auto-encoder +  Classification layer
class ClassifProto(torch.nn.Module):
    """The classification module uses the similarity between the latent data and the prototypes to predict the class.

    Arguments
    ---------
    input_dim : int
        Corresponds to the number of prototypes. This is the similarity between the data in the latent space and each of the prototypes.
    ouput_dim : int
        The number of network classes.
    hidden_size : int
        The neurone number in the hidden layer.
    """
    def __init__(self, input_dim, ouput_dim, hidden_size = 500):
        super(ClassifProto, self).__init__()
        self.fc1 = Linear(in_features=input_dim, out_features=hidden_size)
        self.fc2 = Linear(hidden_size, ouput_dim)

    def forward(self, x):
        # print("classif X", x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
class ProtoVAEBuilder(torch.nn.Module):
    """This module couples the different modules. encoder, prototyping layer, decoder and classification layer.

    Arguments
    ---------
    latent_dims : int
        The dimension of data in latent space. this value will also be used as the dimension of prototypes.
    n_prototypes : int
        The number of prototypes to be learned in the latent space. it must be greater than or equal to the number of classes.
    num_class : int
        The number of classes for the classification layer.
    hidden_size : int 
        The number of neurons in the hidden layer of the classification layer.
    in_channels : int
        Input data size. default 1 for xvector.

    Example
    -------
    >>> inp_tensor = torch.rand([64, 1, 256])
    >>> net = ProtoVAEBuilder(latent_dims=4, n_prototypes=2, num_class=2)
    >>> net.to(device)
    >>> out_tensor, decoded, prototype = net(inp_tensor)
    >>> print("classif output: ", out_tensor.shape, "Decoder output", decoded.shape, "Prototypes: ", prototype.shape)
    >>>
    torch.Size([4, 10, 5])
    """
    def __init__(self, input_shape, latent_dims=4, n_prototypes=2, num_class=2, in_channels=1, hidden_size=50):
        super(ProtoVAEBuilder, self).__init__()
        self.vae = VariationalAutoencoder(latent_dims=latent_dims, n_prototypes=n_prototypes, in_channels=in_channels, input_shape = input_shape)
        self.fcn = ClassifProto(input_dim=n_prototypes, ouput_dim=num_class, hidden_size=hidden_size)

    def forward(self, x, labels): #for this version label is not use
        decoded, prototypes ,latent_sim = self.vae(x)
        predic = self.fcn(latent_sim) 
        return predic, decoded, prototypes
    
class ProtoVAELoss(nn.Module):
    def __init__(self):
        super(ProtoVAELoss, self).__init__()

    """
        kld_weight: float
        model: ProtoVAEBuilder
        The reduction factor of the Kullback-Leibler divergence
    """
    """
        This module implements the loss function of the ProtoVAEBuilder.
    """
    def forward(self, model, prediction, target, input, input_decoded, 
                alpha_R1=0.5, alpha_R2=0.5, alpha_R3=0.02, kld_weight=0.0025):
        """
        Parameters
        ----------
        model : ProtoVAEBuilder
            The model to be trained.
        prediction : torch.tensor
            The output of the classification layer.
        target : torch.tensor
            The target of the classification layer.
        input : torch.tensor
            The input of the autoencoder.
        input_decoded : torch.tensor
            The decoded input of the autoencoder.
        alpha_R1 : float 
            The reduction factor of the R1 loss.
        alpha_R2 : float 
            The reduction factor of the R2 loss.
        alpha_R3 : float 
            The reduction factor of the R3 loss.
        kld_weight : float 
            The reduction factor of the KL divergence loss.

        Returns
        -------
        loss : torch.tensor
            The total loss of the model.
        """
        # The loss of the classification layer
        classif_loss = F.nll_loss(prediction, target)
        # The loss of the autoencoder
        reconst_error = F.mse_loss(input_decoded, input)
        # The KL divergence loss
        kl_loss = model.vae.encoder.kl * kld_weight
        # The R1 loss
        R_1 = model.vae.proto.R_1 * alpha_R1
        # The R2 loss
        R_2 = model.vae.proto.R_2 * alpha_R2 
        # The R3 loss
        R_3 = (1/( model.vae.proto.R_3+ 1)) * alpha_R3
        # The total loss
        loss = classif_loss + reconst_error + kl_loss + R_1 + R_2 + R_3
        # Mask for misclassification on specific class
        # mask = target == 9
        # high_cost = (loss * mask.float()).mean()
        return loss # + high_cost
