import selfies as sf
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from torch import nn
from torch.utils.data import DataLoader
import torch
import argparse
import myModel
import dataloader


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
        
        N_AB = torch.dot(output, label).item()
        N_A = torch.sum(output).item()
        N_B = torch.sum(label).item()
        coeff = N_AB / (N_A + N_B - N_AB)
        loss = 1-coeff
        total_loss += loss
    
    return float(total_loss/len(zipped))




def main(learning_rate=0.01, num_epochs=5):

    print("something")

    LEARNING_RATE = learning_rate
    NUM_EPOCHS = num_epochs

    #Get data
    data = dataloader.Selfies_fp_dataset("all_data.csv")

    train_size = int(len(data) * 0.6)
    val_size = int(len(data) * 0.2)
    test_size = len(data) - val_size - train_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, val_size, test_size])

    train_loader=DataLoader(train_dataset, batch_size=1000, shuffle=True, num_workers=5)
    val_loader=DataLoader(val_dataset, batch_size=1000, shuffle=True, num_workers=5)
    test_loader=DataLoader(test_dataset, batch_size=1000, shuffle=True, num_workers=5)


    print(len(data.vocab_stoi))

    #make model
    input_layer_len = data.max_len * len(data.vocab_stoi)
    net = myModel.Net(input_layer_len, 1600, 1800, 2048)

    #hyperparam
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    #One batch step
    def train_one_epoch():
        running_loss = 0.
        last_loss = 0.

        for i, batch in enumerate(train_loader):
            inputs, labels = batch
            optimizer.zero_grad()       
            outputs = net(inputs)
            lossFxn = nn.BCELoss()
            loss = lossFxn(outputs, labels)
            loss.backward()
            optimizer.step()

            losses = tanimotoLoss(pred_outputs, labels)
            print(losses)

            #reporting performance
            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / 100 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.0
        return last_loss


    for epoch in range(NUM_EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))
        net.train(True)
        avg_loss = train_one_epoch()

        net.train(False)
        
        running_vloss = 0.0
        for i, sample in enumerate(val_loader):
            inputs, labels = sample

            pred_outputs = net(inputs)
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
