"""Train et test function for VAE

Authors
 * St Germes Bengono Obiang 2023
 * Norbert Tsopze 2023
"""
import sys
import torch.nn.functional as F
import torch.optim as optim
import torch
from prototypeVariational import ProtoVAELoss, ProtoVAEBuilder
from ProtoCondVariational import CondVAELoss, CondVAEBuilder
loss_F_Vanilla =  ProtoVAELoss()
loss_F_Cond =  CondVAELoss()
from utils import printProgressBar
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


#Training function definition
"""
    model: ProtoVAEBuilder
"""
def train(model, device, dataloader, optimizer):
    """
    Train the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    device : torch.device
        The device to use for training.
    dataloader : torch.utils.data.DataLoader
        The data loader for the training data.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.

    Returns
    -------
    avg_loss : float
        The average loss of the model after training.
    optimizer : torch.optim.Optimizer
        The optimizer used for training.
    proto : torch.Tensor
        The prototypes of the model.
    """
    model.train()
    train_loss = 0.0
    if isinstance(model, ProtoVAEBuilder):
        loss_F = loss_F_Vanilla
    elif isinstance(model, CondVAEBuilder):
        loss_F = loss_F_Cond
    else:
        print("Model type not support")
        sys.exit(1)
    # Iterate the dataloader
    printProgressBar(0, len(dataloader), prefix = 'Train:', suffix = 'Complete', length = 50)
    i = 1
    proto = None
    for x, y in dataloader:
        # Move tensor to the proper device
        x =  x.to(device)
        predic, decoded, prototypes = model(x, y)
        proto = prototypes
        loss =  loss_F(model, target=y, prediction=predic, input_decoded=decoded, input=x)
        #Start back propoagation
        optimizer.zero_grad() #avoid grad accumulation
        loss.backward() # compute dloss/dx for all parametter 
        optimizer.step() # update all parametter using gradient
        train_loss += loss.item()
        time.sleep(0.1)
        printProgressBar(i, len(dataloader), prefix = 'Train:', suffix = f'Batch loss: {round(train_loss/i,3)}', length = 50)
        i=i+1
    avg_loss = train_loss/ len(dataloader.dataset)
    time.sleep(0.1)
    return avg_loss, optimizer, proto

#Validation function definition
"""
     model: ProtoVAEBuilder
"""
def valid(model, device, dataloader, label_encoder):
    """
    Evaluate the model on a validation set.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be evaluated.
    device : torch.device
        The device to use for evaluation.
    dataloader : torch.utils.data.DataLoader
        The data loader for the validation data.
    label_encoder : LabelEncoder
        The label encoder for the target labels.

    Returns
    -------
    val_loss : float
        The average loss of the model on the validation data.
    acc : float
        The accuracy of the model on the validation data.
    correct_predict : int
        The number of correctly predicted labels.
    labels : dict
        The target labels and the predicted labels.
    """
    model.eval()
    if isinstance(model, ProtoVAEBuilder):
        loss_F = loss_F_Vanilla
    elif isinstance(model, CondVAEBuilder):
        loss_F = loss_F_Cond
    else:
        print("Model type not support")
        sys.exit(1)
    val_loss = 0.0
    correct_predict = 0
    printProgressBar(0, len(dataloader), prefix = 'Valid:', suffix = 'Complete', length = 50)
    i = 1
    target_labels = []
    predict_labels = []
    with torch.no_grad(): # No need to track the gradients
        for x, y in dataloader:
            # Move tensor to the proper device
            x = x.to(device)
            predic, decoded, _ = model(x, y)
            loss =  loss_F(model, target=y, prediction=predic, input_decoded=decoded, input=x)
            val_loss += loss.item()
            pred = predic.data.max(1, keepdim=True)[1]
            correct_predict += pred.eq(y.data.view_as(pred)).sum()
            printProgressBar(i, len(dataloader), prefix = 'Valid:', suffix = f'Batch loss: {round(val_loss/i,3)}', length = 50)
            # Save the target labels and the predicted labels
            target_labels.extend(label_encoder.inverse_transform(y))
            predict_labels.extend(label_encoder.inverse_transform(pred.detach()))
            i=i+1
    acc = correct_predict / len(dataloader.dataset)
    return val_loss / len(dataloader.dataset), acc, correct_predict, {"target": target_labels, "predict": predict_labels}



