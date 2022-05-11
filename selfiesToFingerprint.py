import selfies as sf
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit import RDLogger
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem import DataStructs
from rdkit.Chem.Draw import IPythonConsole
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from functools import partial
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import myModel


def fingerprint_from_smile(smile=str()):
  mol = Chem.MolFromSmiles(smile)
  fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) #could 2048 bits be a hyperparams
  return fp

def selfies_alphabet(selfies_list=list()):

    #  Define selfies Alphabet with padding symbol included 
    all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
    all_selfies_symbols.add('[nop]') # [nop] is the canonical padding symbol supported by the selfies library 
    selfies_alphabet = list(sorted(all_selfies_symbols))

    return selfies_alphabet


def tanimotoLoss(outputs, labels):

    zipped = list(zip(outputs, labels))
    
    total_loss = 0
    for output, label in zipped:
        #specifying the batch size
        #output = output > 0.5
        
        N_AB = torch.dot(output, label)
        N_A = torch.sum(output)
        N_B = torch.sum(label)
        coeff = N_AB / (N_A + N_B - N_AB)
        loss = 1-coeff
        total_loss += loss
    
    return total_loss/len(zipped)




def main(LEARNING_RATE, NUM_EPOCHS):

    #Get data
    train_data = pd.read_csv("moses_train.csv")
    test_data = pd.read_csv("moses_test.csv")
    raw_data = pd.concat([train_data, test_data])

    #get seflie alphabet and maxlen of molecules
    max_len = max(sf.len_selfies(s) for s in raw_data["selfies"])
    alphabet = selfies_alphabet(raw_data["selfies"])
    vocab_stoi = {symbol: idx for idx, symbol in enumerate(alphabet)}
    vocab_itos = {idx: symbol for symbol, idx in vocab_stoi.items()}

    alpha_len = len(vocab_stoi)


    selfies = list(raw_data["selfies"])
    smiles = list(raw_data["smiles"])

    #split into train, validate, test and then zip inputs and targets
    selfies_train, selfies_test, smiles_train, smiles_test = train_test_split(selfies, smiles, test_size=0.33, random_state=42)
    selfies_train, selfies_val, smiles_train, smiles_val = train_test_split(selfies_train, smiles_train, test_size=0.20, random_state=42)
    train_data = list(zip(selfies_train, smiles_train))
    val_data = list(zip(selfies_val, smiles_val))
    test_data = list(zip(selfies_test, smiles_test))

    #construct dataloaders
    train_loader = DataLoader(train_data, batch_size=1000, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1000, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=True)

    #make model
    net = myModel.Net(max_len*alpha_len, 1600, 1800, 2048)

    #hyperparam
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    #One batch step
    def train_one_epoch(epoch_index):
        running_loss = 0.
        last_loss = 0.

        for i, sample in enumerate(train_loader):
            selfies, smiles = sample
            
            inputs = sf.batch_selfies_to_flat_hot(selfies, vocab_stoi, pad_to_len=max_len)
            inputs = torch.tensor(inputs).float()
            
            labels = []
            for smile in smiles:
                fp = fingerprint_from_smile(smile)
                fp = fingerprint_from_smile(smile)
                fp_arr = np.zeros((0,), dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp,fp_arr)
                labels.append(fp_arr)
            labels = torch.tensor(labels)
            
            optimizer.zero_grad()       
            outputs = net(inputs)
            lossFxn = nn.BCELoss()
            loss = lossFxn(outputs, labels)
            loss.backward()
            optimizer.step()


            #reporting performance
            running_loss += loss.item()
            if i % 500 == 499:
                last_loss = running_loss / 500 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.0
        return last_loss


    for epoch in range(NUM_EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))
        net.train(True)
        avg_loss = train_one_epoch(epoch)

        net.train(False)
        
        running_vloss = 0.0
        for i, sample in enumerate(val_loader):
            seflies, smiles = sample

            inputs = sf.batch_selfies_to_flat_hot(selfies, vocab_stoi, pad_to_len=max_len)
            inputs = torch.tensor(inputs).float()
            
            labels = []
            for smile in smiles:
                fp = fingerprint_from_smile(smile)
                fp = fingerprint_from_smile(smile)
                fp_arr = np.zeros((0,), dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp,fp_arr)
                labels.append(fp_arr)
            labels = torch.tensor(labels)

            pred_outputs = net(inputs)
            tuples = list(zip(pred_outputs, labels))
            losses = tanimotoLoss(pred_outputs, labels)
            running_vloss +=losses

        #report epoch loss and save model
        avg_loss = running_vloss/(i+1)
        print("EPOCH " + str(epoch) + " Avg Tanimoto Loss: " + str(avg_loss)) 
        model_path = 'model_{}'.format(epoch)
        torch.save(net.state_dict(), model_path)


if __name__ == '__main__':
    pass

    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default='0.01')
    parser.add_argument('--num_epochs', type=int, default=5)

    args, _ = parser.parse_known_args()

    main(learning_rate=args.learning_rate,
         num_epochs=args.num_epochs)
