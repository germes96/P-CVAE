"""Library implementing prototype based variational auto-encoder

Authors
 * St Germes Bengono Obiang 2023
 * Norbert Tsopze 2023
"""
import pkg_resources
import sys
import datetime
import pandas as pd
import pip
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from csv import DictReader
from sklearn.manifold import TSNE
import plotly.express as px
from tqdm import tqdm

def str2List(values):
    """
    Convert a list of string containing a list of float separated by space
    to a numpy array.

    Parameters
    ----------
    values : list of str
        A list of string containing a list of float separated by space.

    Returns
    -------
    numpy array
        A numpy array containing the list of float from the string.
    """
    result = []
    for value in values:
        # Remove the [] from the string
        nlist = value.strip("[]").split()
        # Convert the string to float
        nlist = [float(x) for x in nlist]
        # Add the list to the result
        result.append(nlist)
    # Convert the result to numpy array
    return np.array(result)

def loadCSVDataset(path, batch_size=256, validation_plit=False, valid_size=0.1, x_vector_field_name = "embedding", label_field= "label"):
    """
    Load a csv dataset and return a data loader.

    Parameters
    ----------
    path : str
        The path to the csv file.
    batch_size : int, optional
        The batch size of the data loader. The default is 256.
    validation_plit : bool, optional
        If True, split the dataset into a training set and a validation set.
        The default is False.
    valid_size : float, optional
        The proportion of the dataset to use for the validation set.
        The default is 0.1.
    x_vector_field_name : str, optional
        The name of the column containing the vector data.
        The default is "embedding".
    label_field : str, optional
        The name of the column containing the labels.
        The default is "label".

    Returns
    -------
    main_loader : DataLoader
        The data loader for the main dataset.
    valid_loader : DataLoader
        The data loader for the validation set.
    data_shape : list
        The shape of the data.
    label_number : int
        The number of labels.
    encoder : LabelEncoder
        The label encoder.
    """
    encoder = LabelEncoder()
    data = pd.read_csv(path, converters={f"{x_vector_field_name}": lambda x: x.replace('\n', '').strip()})
    X = str2List(data[f"{x_vector_field_name}"].to_numpy()) 
    label_number = data[f'{label_field}'].nunique()
    y = data[f"{label_field}"].to_numpy()
    encoder.fit(y)
    y = encoder.transform(y)

    #################################### Convert 2tensor data ###########################################
    X = torch.tensor(X, dtype=torch.float32)
    X =  X.reshape(X.size()[0], 1, X.size()[1])
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    y = torch.flatten(y).type(torch.LongTensor)

    data_shape = list(X[0].shape)
    print("Data shape", data_shape)
    data_shape.insert(0, batch_size)

    if validation_plit:
        # split datset
        X_main, X_valid, y_main, y_valid = train_test_split(X, y, train_size=(1-valid_size), shuffle=True)
        # create data loader
        main_loader = DataLoader(list(zip(X_main, y_main)), shuffle=True, batch_size=batch_size)
        valid_loader = DataLoader(list(zip(X_valid,y_valid)), batch_size=batch_size)
        return main_loader, valid_loader, data_shape, label_number, encoder
    else:
        # create data loader
        main_loader = DataLoader(list(zip(X, y)), shuffle=True, batch_size=batch_size)
        return main_loader, data_shape, label_number, encoder
    
def loadPKLDataset(path, batch_size=256, validation_plit=False, valid_size=0.1, embedding_field = "embedding", label_field= "label"):
    """
    Load a dataset from a pickle file and return a data loader.

    Parameters
    ----------
    path : str
        The path to the pickle file.
    batch_size : int, optional
        The batch size of the data loader. The default is 256.
    validation_plit : bool, optional
        If True, split the dataset into a training set and a validation set.
        The default is False.
    valid_size : float, optional
        The proportion of the dataset to use for the validation set.
        The default is 0.1.
    embedding_field : str, optional
        The name of the column containing the vector data.
        The default is "embedding".
    label_field : str, optional
        The name of the column containing the labels.
        The default is "label".

    Returns
    -------
    main_loader : DataLoader
        The data loader for the main dataset.
    valid_loader : DataLoader
        The data loader for the validation set.
    data_shape : list
        The shape of the data.
    label_number : int
        The number of labels.
    encoder : LabelEncoder
        The label encoder.
    """
    encoder = LabelEncoder()
    with open(path, 'rb') as f:
        df = pd.read_pickle(f)
        label_number = df[f'{label_field}'].nunique()
        X = df[f"{embedding_field}"].to_numpy()
        X = np.vstack(X).astype(np.float32)
        y = df[f'{label_field}'].to_numpy()
        encoder.fit(y)
        y = encoder.transform(y)
        print(y[200])

        # Convert to tensor data
        X = torch.tensor(X, dtype=torch.float32)
        X =  X.reshape(X.size()[0], 1, X.size()[1])
        y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        y = torch.flatten(y).type(torch.LongTensor)
        data_shape = list(X[0].shape)
        data_shape.insert(0, batch_size)
        print("DATA SHAPE", data_shape)

        if validation_plit:
            # Split dataset
            X_main, X_valid, y_main, y_valid = train_test_split(X, y, train_size=(1-valid_size), shuffle=True)
            # Create data loader
            main_loader = DataLoader(list(zip(X_main, y_main)), shuffle=True, batch_size=batch_size)
            valid_loader = DataLoader(list(zip(X_valid,y_valid)), batch_size=batch_size)
            return main_loader, valid_loader, data_shape, label_number, encoder 
        else:
            # Create data loader
            main_loader = DataLoader(list(zip(X, y)), shuffle=True, batch_size=batch_size)
            return main_loader, data_shape, label_number, encoder
        


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# save all execution dependancy
def savepackage(path):
    """
    Save all packages and their version in a text file env.log
    in the given directory.
    """
    # Get all packages and their version
    dists = [str(d).replace(" ",  "==") for d in pkg_resources.working_set]
    # Create env.log file
    with open(f'{path}/env.log', 'w') as log:
        # Write header
        log.write('VAE-C Environement Description\n==============================\n')
        # Write python version
        log.write(f'Python version:\n{sys.version} ({datetime.datetime.now()})\n')
        # Write separator
        log.write("==============================\n")
        # Write packages and their version
        for pack in dists:  
            log.write(f"{pack}\n")

def saveTrainLog(path, log):
    """
    Save the given log to a file named "train_log.log" in the given path.
    
    Parameters
    ----------
    path : str
        The path where the log file should be saved.
    log : str
        The log to save.
    """
    with open(f'{path}/train_log.log', 'a') as file:
        # Write the log to the file
        file.write(f'{log}\n')

def saveTrainStatsCSV(path, stats, file_name="train_stats.csv", open_option="a+"):
    """
    Save the traning stats to a CSV file named `file_name` in the given `path`.
    
    Parameters
    ----------
    path : str
        The path where the CSV file should be saved.
    stats : list of dict
        The stats to save.
    file_name : str, optional
        The name of the CSV file. Default is "train_stats.csv".
    open_option : str, optional
        The option to open the file. Default is "a+" which means append.
    
    Notes
    -----
    If the file does not exist, it will be created. If the file already exists, the
    data will be appended to the file.
    """
    file_exist = False
    if os.path.exists(f'{path}/{file_name}'):
        file_exist = True
    with open(f'{path}/{file_name}', open_option) as f:
        # Write all the dictionary keys in a file with commas separated.
        if not file_exist:
            f.write(','.join(stats[0].keys()))
            f.write('\n') # Add a new line
        for row in stats:
            # Write the values in a row.
            f.write(','.join(str(x) for x in row.values()))
            f.write('\n') # Add a new line

def loadCSV(path):
    """
    Load the first row of a CSV file into a dictionary.
    
    Parameters
    ----------
    path : str
        The path to the CSV file.
    
    Returns
    -------
    dict
        The first row of the CSV file loaded into a dictionary.
    """
    with open(path, 'r') as f:    
        dict_reader = DictReader(f)
        list_of_dict = list(dict_reader)
        # Return only the first row of the CSV file.
        return list_of_dict[0]
    
def classification_report_csv(report, path):
    """
    Save the classification report to a CSV file.
    
    Parameters
    ----------
    report : str
        The classification report as a string.
    path : str
        The path to the directory where the CSV file will be saved.
    
    Returns
    -------
    None
    """
    report_to_df = classification_report_to_dataframe(report)
    report_to_df.to_csv(f'{path}/classification_report.csv', index = False)

def classification_report_to_dataframe(str_representation_of_report):
    """
    Convert the classification report from a string to a pandas DataFrame.

    Parameters
    ----------
    str_representation_of_report : str
        The classification report as a string.

    Returns
    -------
    pd.DataFrame
        The classification report converted to a pandas DataFrame.
    """
    # Split the string into a table where each row is a list of values
    split_string = [x.split(' ') for x in str_representation_of_report.split('\n')]
    # Get the column names from the first row of the table
    column_names = ['']+[x for x in split_string[0] if x!='']
    # Initialize an empty list to store the rows of the DataFrame
    values = []
    # Iterate through the table and add each row to the DataFrame
    for table_row in split_string[1:-1]:
        table_row = [value for value in table_row if value!='']
        if table_row!=[]:
            values.append(table_row)
    # Iterate through the DataFrame and add the 'avg' and 'total' rows
    for i in values:
        for j in range(len(i)):
            # Check if the value is 'avg'
            if i[1] == 'avg':
                # Join the first two values with a space
                i[0:2] = [' '.join(i[0:2])]
            # Check if the length of the row is 3
            if len(i) == 3:
                # Insert NaN values in the second and third positions
                i.insert(1,np.nan)
                i.insert(2, np.nan)
            else:
                pass
    # Create the DataFrame
    report_to_df = pd.DataFrame(data=values, columns=column_names)
    # Return the DataFrame
    return report_to_df

def saveLatentSpace(model, dataloader, label_encoder, device, path):
    """
    Save the latent space of the model to a CSV file.

    Parameters
    ----------
    model : torch.nn.Module
        The model to save the latent space of.
    dataloader : torch.utils.data.DataLoader
        The dataloader for the data set.
    label_encoder : sklearn.preprocessing.LabelEncoder
        The label encoder for the target labels.
    device : torch.device
        The device to use for the computation.
    path : str
        The path to save the CSV file to.
    """
    with open(f'{path}/latent_projections.csv', "w+") as f:
        # Write the header line
        f.write('label, latent')
        f.write('\n') # Add a new line
        # Iterate through the data set and write each sample to the CSV file
        for sample in tqdm(dataloader.dataset, desc="SAVE LATENT PROJECTION"):
            input = sample[0].unsqueeze(0).to(device)
            label = sample[1].tolist()
            label = label_encoder.inverse_transform([label])
            with torch.no_grad():
                encoded_input  = model.vae.encoder(input, sample[1])
            #Write data_on csv
            f.write(f'{label[0]}, {encoded_input.detach()[0].numpy()}')
            f.write('\n') # Add a new line
            

def TSEVisualization(dataloader, model ,Projector, device, path, type="test"):
    """
    Train TSNE VISUALIZATION FOR THE DATASET

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        The dataloader for the data set.
    model : torch.nn.Module
        The model to use for the TSNE visualization.
    Projector : torch.nn.Module
        The projector to use for the TSNE visualization.
    device : torch.device
        The device to use for the computation.
    path : str
        The path to save the CSV file to.
    type : str
        The type of the visualization. Default is 'test'.
    """
    save_folder = f"{path}/visualization"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    model.eval()
    encoded_samples = []
    print(f"TRAIN TSNE VISUALIZATION FOR {type}")
    #add  each data in dataset on encoded list
    for sample in tqdm(dataloader.dataset):
        input = sample[0].unsqueeze(0).to(device)
        label = sample[1]
        # Encode image
        with torch.no_grad():
            encoded_input  = model.vae.encoder(input, label)
        # Append to list
        encoded_input = encoded_input.flatten().cpu().numpy()
        encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_input)}
        encoded_sample['label'] = label
        encoded_samples.append(encoded_sample)
    #add  each prototype on encoded list
    if hasattr(model.vae, 'proto'):
        prototypes = model.vae.proto.prototypes.detach()
    elif hasattr(model.vae.encoder, 'mean_class'):
         prototypes = model.vae.encoder.mean_class.detach()
    if prototypes != None:
        visualisation_size = np.full(shape=len(encoded_samples)+prototypes.shape[0],fill_value=1,dtype=np.int) # define de size of each dot on the latent visualisation plot
        for j, proto in enumerate(prototypes):
            encoded_input = proto.flatten().cpu().numpy()
            encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_input)}
            encoded_sample['label'] = f"proto {j}"
            encoded_samples.append(encoded_sample)
            visualisation_size[-(j+1)] = 4 #proto size more bigger that others
    else:
        visualisation_size = np.full(shape=len(encoded_samples),fill_value=1,dtype=np.int) # define de size of each dot on the latent visualisation plot
    # Transform encoded list into dataframe    
    encoded_samples = pd.DataFrame(encoded_samples)
    #Train TSNE VISILIZATION
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(encoded_samples.drop(['label'],axis=1))
    print(f"GENERATE TSNE VISUALIZATION FOR {type}")
    fig = px.scatter(tsne_results, x=0, y=1, color=encoded_samples.label.astype(str) ,labels={'0': 'tsne-2d-one', '1': 'tsne-2d-two'}, size=visualisation_size)
    print(f"SAVE TSNE VISUALIZATION FOR {type}")
    # fig.show()
    fig.write_image(f"{save_folder}/latent_tsne_{type}.png", width=1920, height=1080, scale=3)
    fig.write_html(f"{save_folder}/latent_tsne_{type}.html")



# main_loader, valid_loader, data_shape, label_number, encoder = loadCSVDataset(path="dataset/VoxCeleb/emo_meta2.csv", validation_plit=True)
# main_loader, valid_loader, data_shape, label_number, encoder = loadCSVDataset(path="dataset/Gender/ALLIES/gender_x-vector_train.csv", validation_plit=True)
# print(label_number)
# print(data_shape)
# print(len(main_loader.dataset))


