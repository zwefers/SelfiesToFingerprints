from torch.utils.data import Dataset
import pandas as pd
import selfies as sf
from rdkit.Chem import DataStructs, MolFromSmiles, AllChem
import numpy as np
from torch import tensor, flatten
import torch.nn.functional as F




class Selfies_fp_dataset(Dataset):

    def __init__(self, data_path):

        #loading data
        self.raw_data = pd.read_csv(data_path)

        #making alphabet
        selfies = self.raw_data ["selfies"]
        alphabet = selfies_alphabet(selfies)

        #saving attributes
        self.max_len = max(sf.len_selfies(s) for s in self.raw_data["selfies"])
        self.vocab_stoi = {symbol: idx for idx, symbol in enumerate(alphabet)}

    def __len__(self):
        return self.raw_data.shape[0]
    
    def __getitem__(self, idx):
        selfie = self.raw_data["selfies"][idx]
        smile = self.raw_data["smiles"][idx]

        #process selfie
        selfies_split = list(sf.split_selfies(selfie))
        selfies_split  = selfies_split + ['[nop]' for i in range(self.max_len - len(selfies_split))]
        selfie_integers = tensor([self.vocab_stoi[symbol] for symbol in selfies_split])
        one_hot_selfie=F.one_hot(selfie_integers, num_classes=len(self.vocab_stoi)).float() 
        one_hot_selfie = flatten(one_hot_selfie)


        #process smile
        fp = fingerprint_from_smile(smile)
        fp_arr = np.zeros((0,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp,fp_arr)
        fp = tensor(fp_arr)

        return one_hot_selfie, fp


def fingerprint_from_smile(smile=str()):
    mol = MolFromSmiles(smile)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) #could 2048 bits be a hyperparams
    return fp

def selfies_alphabet(selfies_list=list()):

    #  Define selfies Alphabet with padding symbol included 
    all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
    all_selfies_symbols.add('[nop]') # [nop] is the canonical padding symbol supported by the selfies library 
    selfies_alphabet = list(sorted(all_selfies_symbols))

    return selfies_alphabet