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

def str2List(values):
    result = []
    for value in values:
        nlist = value.strip("[]").split()
        nlist = [float(x) for x in nlist]
        result.append(nlist)
    return np.array(result)

def loadCSVDataset(path, batch_size=256, validation_plit=False, valid_size=0.1, x_vector_field_name = "embedding", label_field= "label"):
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
    # print("Data shape", data_shape)
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

        #################################### Convert 2tensor data ###########################################
        X = torch.tensor(X, dtype=torch.float32)
        X =  X.reshape(X.size()[0], 1, X.size()[1])
        y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        y = torch.flatten(y).type(torch.LongTensor)
        data_shape = list(X[0].shape)
        data_shape.insert(0, batch_size)
        print("DATA SHAPE", data_shape)

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
    dists = [str(d).replace(" ",  "==") for d in pkg_resources.working_set]
    # Create env.log file
    with open(f'{path}/env.log', 'w') as log:
        log.write('VAE-C Environement Description\n==============================\n')
        log.write(f'Python version:\n{sys.version} ({datetime.datetime.now()})\n==============================\n')
        for pack in dists:  log.write(f"{pack}\n")

def saveTrainLog(path, log):
    with open(f'{path}/train_log.log', 'a') as file:
        file.write(f'{log}\n')

def saveTrainStatsCSV(path, stats, file_name = "train_stats.csv", open_option="a+"):
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
    with open(path, 'r') as f:    
        dict_reader = DictReader(f)
        list_of_dict = list(dict_reader)
        return list_of_dict[0]
    
def classification_report_csv(report, path):
    report_to_df = classification_report_to_dataframe(report)
    report_to_df.to_csv(f'{path}/classification_report.csv', index = False)

def classification_report_to_dataframe(str_representation_of_report):
    split_string = [x.split(' ') for x in str_representation_of_report.split('\n')]
    column_names = ['']+[x for x in split_string[0] if x!='']
    values = []
    for table_row in split_string[1:-1]:
        table_row = [value for value in table_row if value!='']
        if table_row!=[]:
            values.append(table_row)
    for i in values:
        for j in range(len(i)):
            if i[1] == 'avg':
                i[0:2] = [' '.join(i[0:2])]
            if len(i) == 3:
                i.insert(1,np.nan)
                i.insert(2, np.nan)
            else:
                pass
    report_to_df = pd.DataFrame(data=values, columns=column_names)
    return report_to_df


