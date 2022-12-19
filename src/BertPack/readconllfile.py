import pandas as pd
import numpy as np
import os
def read_conll_file(adress)-> pd.DataFrame:
    '''
    Renvoyer un Dataframe columns=['Seqs', 'Skills', 'Knowledges']
    '''
    sequnces = [] #indice0
    Skills = [] #indice1
    Knowledges = [] #indice2
    
    with open(adress,'r') as cf:
        seq = ""    
        ski = ""
        Know = ""
        for line in cf.readlines():
            
            si = line.strip().split('\t')
            if "" in si:
                si = [x for x in si if x != ""]
            
            if len(si) < 3 :
                if len(seq) > 0:
                    sequnces.append(seq)
                    Skills.append(ski)
                    Knowledges.append(Know)
                    #print(seq + ";", ski+";", Know)
                    seq = ""    
                    ski = ""
                    Know = ""
                continue
            else:    
                seq = seq + si[0] + " "
                ski = ski + si[1] + " "
                Know = Know + si[2] + " "
    df = pd.DataFrame({'Seqs':sequnces, 'Skills':Skills, 'Knowledges':Knowledges})
    return df

def read_all_conll_file(path, filter):
    files = os.listdir(path)
    dflist = []
    for file in files:
        if not os.path.isdir(file) and filter in file:
            df = read_conll_file(path+"/"+file)
            dflist.append(df)
    return pd.concat(dflist)

if __name__ == "__main__":
    dossierchemin = "./data/Skills/"
    read_conll_file(dossierchemin+"data_scientist.txt")
    read_conll_file(dossierchemin+"skillspan_house_dev.conll")
    read_all_conll_file(dossierchemin, "train")