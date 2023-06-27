"""Train et test function for VAE

Authors
 * St Germes Bengono Obiang 2023
 * Norbert Tsopze 2023
"""
import torch.nn.functional as F
import torch.optim as optim
import torch
from prototypeVariational import ProtoVAELoss
loss_F =  ProtoVAELoss()
from utils import printProgressBar
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


#Training function definition
"""
    model: ProtoVAEBuilder
"""
def train(model, device, dataloader, optimizer):
    model.train()
    train_loss = 0.0
    # Iterate the dataloader
    printProgressBar(0, len(dataloader), prefix = 'Train:', suffix = 'Complete', length = 50)
    i = 1
    proto = None
    for x, y in dataloader:
        # Move tensor to the proper device
        x =  x.to(device)
        predic, decoded, prototypes = model(x)
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
    model.eval()
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
            predic, decoded, _ = model(x)
            loss =  loss_F(model, target=y, prediction=predic, input_decoded=decoded, input=x)
            val_loss += loss.item()
            pred = predic.data.max(1, keepdim=True)[1]
            correct_predict += pred.eq(y.data.view_as(pred)).sum()
            printProgressBar(i, len(dataloader), prefix = 'Valid:', suffix = f'Batch loss: {round(val_loss/i,3)}', length = 50)
            # print(label_encoder.inverse_transform(y), "  -> " ,label_encoder.inverse_transform(pred))
            # print(y.tolist())
            # labels.extend(y.tolist())
            target_labels.extend(label_encoder.inverse_transform(y))
            predict_labels.extend(label_encoder.inverse_transform(pred.detach()))
            i=i+1
    acc = correct_predict / len(dataloader.dataset)
    return val_loss / len(dataloader.dataset), acc, correct_predict, {"target": target_labels, "predict": predict_labels}



