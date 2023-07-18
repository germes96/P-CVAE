"""
DOTO

- load dataset: done
- create model: done
- Train model: done
- save model: done
- load model
- evaluate model
- display data
"""
import uuid
import datetime
import os
import sys
import argparse

import torch
import utils
import prototypeVariational as VAE
import VariationalConditional as CondVAE
from sklearn.metrics import classification_report
import TrainUtils

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
save_path="models/"



def load_checkpoint(model, optimizer=None, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer != None: 
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch


def train_model(model, train_loader, valid_loader, label_encoder, epoch_number=50, lr=1e-3, seed = 0):
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    start_epoch = 0
    # if checkpoint exist, load. else create dir
    if os.path.exists(f"{save_path}{seed}"): #create dire
       model, optim, start_epoch = load_checkpoint(model=model, optimizer=optim, filename=f"{save_path}{seed}/checkpoint.pth.tar")
    else: #load old model
        os.makedirs(f"{save_path}{seed}")

    old_acc = 0
    best_epochs = 1
    # Dictionary to save stats
    # stats= {
    #     "train_loss": [],
    #     "valid_loss": [],
    #     "valid_acc": [],
    #     "best_epochs": 1
    # }

    stats = []
    best_epochs = 1

    for epoch in range(epoch_number):
        print(f"EPOCH: ", epoch + start_epoch +1, f"/{epoch_number+ start_epoch}")

        #train and valid
        train_loss, optimizer, prototypes = TrainUtils.train(model,device,train_loader,optim)
        val_loss, acc, correct_predict, _ = TrainUtils.valid(model,device,valid_loader, label_encoder=label_encoder)

        # Save best model
        if(acc>=old_acc):
            print("SAVE MODEL")
            old_acc = acc
            best_epochs =  epoch +  start_epoch + 1
            state = {'epoch': epoch + start_epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict() }
            torch.save(state, f"{save_path}{seed}/checkpoint.pth.tar")
            # torch.save(model.state_dict(), f"{save_path}{save_folder}/checkpoint.pth.tar")

         # Save stats
        stats.append({
             "train_loss": train_loss,
             "valid_loss": val_loss,
             "valid_acc": acc.item(),
             "best_epochs": best_epochs
        })
        #display epoch info:
        log = f"epoch: {epoch + start_epoch +1}, lr: {optim.param_groups[0]['lr']} - train loss: {train_loss} - valid loss: {val_loss}, valid ACC: {100. * correct_predict / len(valid_loader.dataset)}"
        print(log)
        utils.saveTrainLog(path=f"{save_path}{seed}", log=log)
    #save prototype``
    decoded_proto = model.vae.decoder(prototypes).detach().tolist()
    proto = []
    for i in range(len(prototypes.detach().tolist())):
        p = prototypes.detach().tolist()[i]
        proto.append({
            "id": i,
            "latent": str(p).replace(",",""),
            "decoded": str(decoded_proto[i][0]).replace(",",""),
        })
    utils.savepackage(path=f"{save_path}{seed}")
    utils.saveTrainStatsCSV(path=f"{save_path}{seed}", stats=stats)
    utils.saveTrainStatsCSV(path=f"{save_path}{seed}", stats=proto, file_name="prototype.csv", open_option='w')
    return model, stats

def test_model(model, test_loader, label_encoder, seed = 0):
    start_epoch = 0
    # if checkpoint exist, load. else create dir
    if os.path.exists(f"{save_path}{seed}"):  # load old model
       model, optim, start_epoch = load_checkpoint(model=model, filename=f"{save_path}{seed}/checkpoint.pth.tar")
    else:  #create dire
        os.makedirs(f"{save_path}{seed}")

    test_loss, acc, correct_predict, labels = TrainUtils.valid(model, device, test_loader, label_encoder=label_encoder)
    stats = []
    stats.append({
        "test_loss": test_loss,
        "acc": acc.item(),
        "correct_predict": correct_predict
    })
    log = f"epoch: {start_epoch}, test loss: {test_loss} - accuracy: {acc.item()}, correct_predict: {100. * correct_predict / len(test_loader.dataset)}"
    print(log)
    return model, stats, labels


torch.manual_seed(0)
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Prototyped-VAE-C on embeddind data")
    parser.add_argument("-d", dest="dataset", help="The path to the dataset used. Supported formats are pickel and csv. ", type=str, default="dataset/Gender/ALLIES/gender_x-vector_train.csv")
    parser.add_argument("-e", dest="ext", help="specify the dataset format you have chosen. (pkl or csv)", type=str, default="csv")
    parser.add_argument("-s", dest="seed", help="the number of the training unit in which the results will be stored", type=int, default=100)
    parser.add_argument("-n", dest="num_epoch", help="number of training epoch", type=int, default=50)
    parser.add_argument("-t", dest="type", help="Allows you to specify whether it's a training session or a test (train or test).", type=str, default="train")
    parser.add_argument("-l", dest="latent_dim", help="Latent space dimension.", type=int, default=4)
    parser.add_argument("-p", dest="prototype_number", help="Latent space dimension. -1 if you want this value to be equal to the number of classes in the dataset", type=int, default=-1)
    parser.add_argument("--lr", dest="learning_rate", help="The learning rate in the optimization", type=float, default=1e-3)
    parser.add_argument("--train-projection", dest="train_projection", help="if a projection for training data needs to be generated (y/n)", type=str, default="n")
    parser.add_argument("--vae-type", dest="vae_type", help="We implement 2 type of variational encoder network (v/c) v for vanilla and c for conditional", type=str, default="v")
    args = parser.parse_args()
    #Load params
    seed = args.seed
    print("LOAD DATASET")
    if args.type == "train":
        if args.ext == "csv":
            train_loader, valid_loader, input_shape, label_nb, encoder = utils.loadCSVDataset(path=args.dataset, validation_plit=True)
        elif args.ext == "pkl":
            train_loader, valid_loader, input_shape, label_nb, encoder = utils.loadPKLDataset(path=args.dataset, validation_plit=True)
        else:
            print("Dataset extension not support")
            sys.exit(1)
        #Define model hyperparams

        print("LABEL NUMBER", label_nb)
        latent_dims = args.latent_dim
        proto_number = label_nb
        if args.prototype_number != -1:
            proto_number = args.prototype_number
        class_number = label_nb
        epoch_number = args.num_epoch
        lr = args.learning_rate
        # Create model
        if args.vae_type == "v":
            model = VAE.ProtoVAEBuilder(latent_dims=latent_dims, n_prototypes=proto_number, num_class=class_number, input_shape=input_shape)
        elif args.vae_type == "c":
            model = CondVAE.CondVAEBuilder(latent_dims=latent_dims, num_class=class_number, input_shape=input_shape)
        else:
            print("model type not support")
            sys.exit(1)
        # Create execution instance folder
        if not os.path.exists(f"{save_path}{seed}"): 
            os.makedirs(f"{save_path}{seed}")
        params_to_save = [{"latent_dims": latent_dims, "n_prototypes": proto_number, "num_class": class_number, "input_shape": input_shape}]
        utils.saveTrainStatsCSV(path=f"{save_path}{seed}", stats=params_to_save, file_name="arg.csv")
        model.to(device)
        print(model)
        # Train model
        model, stats = train_model(model, train_loader, valid_loader, label_encoder=encoder, epoch_number=epoch_number, seed=seed, lr=lr)
        if args.train_projection == "y":
            utils.TSEVisualization(dataloader=train_loader, model=model, Projector=model.vae.encoder, device=device, path=f"{save_path}{seed}", type="train")
        utils.TSEVisualization(dataloader=valid_loader, model=model, Projector=model.vae.encoder, device=device, path=f"{save_path}{seed}", type="valid")

    elif args.type ==  "test":
        if args.ext == "csv":
            test_loader, input_shape, label_nb, encoder = utils.loadCSVDataset(path=args.dataset, validation_plit=False)
        elif args.ext == "pkl":
            test_loader, input_shape, label_nb, encoder = utils.loadPKLDataset(path=args.dataset, validation_plit=False)
        else:
            print("Dataset extension not support")
            sys.exit(1)
        #load all hyperparametters
        print("START TEST")
        params = utils.loadCSV(path=f"{save_path}{seed}/arg.csv")
        print("LIST OF LOADED PARAMS", params)
        if args.vae_type == "v":
            model = VAE.ProtoVAEBuilder(latent_dims=int(params["latent_dims"]), n_prototypes=int(params["n_prototypes"]), num_class=int(params["num_class"]), input_shape=input_shape)
        elif args.vae_type == "c":
            model = CondVAE.CondVAEBuilder(latent_dims=int(params["latent_dims"]), num_class=int(params["num_class"]), input_shape=input_shape)
        else:
            print("Model type not support")
            sys.exit(1)
        print(model)
        # START TEST ON DATA
        model, stats, labels = test_model(model, test_loader, label_encoder=encoder, seed=seed)
        report = classification_report(labels['target'], labels['predict'])
        print(f"\n\n{report}")
        utils.classification_report_csv(path=f"{save_path}{seed}", report=report)
        utils.saveLatentSpace(dataloader=test_loader, model=model, label_encoder=encoder, device=device, path=f"{save_path}{seed}")
        utils.TSEVisualization(dataloader=test_loader, model=model, Projector=model.vae.encoder, device=device, path=f"{save_path}{seed}")
    # print(stats)