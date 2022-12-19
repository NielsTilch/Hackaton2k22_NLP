import torch.utils.data as t_data
import torch.nn.functional as F
from BertPack.readconllfile import read_conll_file as rcf
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

#Labels
label_list = set([
    "O",
    "B-Knowledge",
    "I-Knowledge",
    "B-Skill",
    "I-Skill"
])
labels_to_ids = {k: v for v, k in enumerate(sorted(label_list))}
ids_to_labels = {v: k for v, k in enumerate(sorted(label_list))}

#dataset class

class SequencesData(t_data.Dataset):
    def __init__(self,df,tokenizer,max_length = 128, flag = 1) -> None:
        '''
        flag = 1 utiliser label skills
        '''
        #'Seqs', 'Skills', 'Knowledges'
        super().__init__()
        txt = df['Seqs']
        sks = [i.split() for i in df['Skills'].values.tolist()]
        kns = [i.split() for i in df['Knowledges'].values.tolist()]
        text = [tokenizer(str(i),
                               padding='max_length', max_length = max_length, 
                                truncation=True, return_tensors="pt") for i in txt]
        self.texts = txt.tolist()
        self.labels_sks = [aligner_les_labels(i,j) for i,j in zip(text, sks)]
        self.labels_kns = [aligner_les_labels(i,j) for i,j in zip(text, kns)]
        self.flag = flag
    def __len__(self):
        if self.flag == 1:
            return len(self.labels_sks)
        else:
            return len(self.labels_kns)
    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_labels(self, idx):
        if self.flag == 1:
            s = self.labels_sks[idx]
            k = self.labels_kns[idx]
            len_s = len(s)
            res = torch.zeros(len_s,5)
            for i in range(len_s):
                if s[i] == -100:
                    res[i] = -100
                else:
                    res[i][s[i]] = 1
                    res[i][k[i]] = 1
            return res
        elif self.flag == 2:
            return torch.LongTensor(self.labels_sks[idx])
        else:           
            return torch.LongTensor(self.labels_kns[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_data, batch_labels

#Fonnctions
def aligner_les_labels(token_input, input_s_labels):
    
    
    
    word_ids = token_input.word_ids()
    labels_for_token_input = []
    previous_word_id = None
    previous_label = None
    for w_idx in word_ids:
        if w_idx is None:
            labels_for_token_input.append(4)
        elif w_idx != previous_word_id:
            try :
                labels_for_token_input.append(labels_to_ids[input_s_labels[w_idx]])
                previous_label = labels_to_ids[input_s_labels[w_idx]]
            except:
                labels_for_token_input.append(4) 
                previous_label = 4
        else:
            labels_for_token_input.append(previous_label)
        previous_word_id = w_idx
    return labels_for_token_input

def train(tokenizer, model, train_dataloader, optimizer, epochs:int, num_sample:int):
    '''
    il faut rappeler Dataloader, ex: train_dataloader = Dataloader(Dataset(df_train))
    
    '''
    #Voir si il y a gpu disponible
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()
    for i in range(epochs):
        f1_B_s= 0
        f1_B_k = 0
        f1_I_s = 0
        f1_I_k = 0
        f1_macro = 0
        loss_val = 0
        model.train()
        tot = 0
        for train_data, train_label in tqdm(train_dataloader):
            max_length = len(train_label[0])
            train_label = train_label.to(device)
            train_data = list(train_data)
            token_input = tokenizer(train_data,
                               padding='max_length', max_length = max_length, 
                                truncation=True, return_tensors="pt").to(device)

            optimizer.zero_grad()
            
            output = model(**token_input, labels = train_label)
            loss = output['loss']
            logits = output['logits']
            logits_clean = logits
            label_clean = train_label
            diction = None
            if use_cuda:
                diction = compute_metrics((logits_clean.detach().cpu().numpy(),label_clean.detach().cpu().numpy()))
            else:
                diction = compute_metrics((logits_clean.detach().numpy(),label_clean.detach().numpy()))
            f1_B_k += diction['f1_B-Knowledge']
            f1_B_s += diction['f1_B-Skill']
            f1_I_k += diction['f1_I-Knowledge']
            f1_I_s += diction['f1_I-Skill']
            f1_macro += diction['macro_f1']
            loss_val += loss.item()
            loss.backward()
            optimizer.step()

            tot += 1
        print(f'''Epochs: {i + 1} | 
                    Loss: {loss_val/tot : .3f} | 
                    F1_score_B-Knowledge: {f1_B_k/tot : .3f} |
                    F1_score_B-Skill: {f1_B_s/tot : .3f} |
                    F1_score_I-Knowledge: {f1_I_k/tot : .3f} |
                    F1_score_I-Skill: {f1_I_s/tot : .3f} |
                    F1_score_Macro: {f1_macro/tot: .3f} |''')
        
def validation(tokenizer, model, dev_dataloader, num_sample:int):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()
    model.eval()
    f1_B_s= 0
    f1_B_k = 0
    f1_I_s = 0
    f1_I_k = 0
    f1_macro = 0
    loss_val = 0
    tot = 0
    for dev_data, dev_label in dev_dataloader:
        max_length = len(dev_label[0])
        dev_label = dev_label.to(device)
        dev_data = list(dev_data)
        token_input = tokenizer(dev_data,
                               padding='max_length', max_length = max_length, 
                                truncation=True, return_tensors="pt").to(device)
        
        
        output = model(**token_input, labels = dev_label)
        loss = output['loss']
        logits = output['logits']
        logits_clean = logits
        label_clean = dev_label
        if use_cuda:
            diction = compute_metrics((logits_clean.detach().cpu().numpy(),label_clean.detach().cpu().numpy()))
        else:
            diction = compute_metrics((logits_clean.detach().numpy(),label_clean.detach().numpy()))
        f1_B_k += diction['f1_B-Knowledge']
        f1_B_s += diction['f1_B-Skill']
        f1_I_k += diction['f1_I-Knowledge']
        f1_I_s += diction['f1_I-Skill']
        f1_macro += diction['macro_f1']
        loss_val += loss.item()
        tot += 1
    print(f''' 
                Loss: {loss_val / tot: .3f} | 
                F1_score_B-Knowledge: {f1_B_k / tot: .3f} |
                F1_score_B-Skill: {f1_B_s / tot: .3f} |
                F1_score_I-Knowledge: {f1_I_k / tot: .3f} |
                F1_score_I-Skill: {f1_I_s / tot: .3f} |
                F1_score_Macro: {f1_macro / tot: .3f} |''')
    
    
    
    

n_labels = 5

def divide(a: int, b: int):
    return a / b if b > 0 else 0

def compute_metrics(p):
    """
    Customize the `compute_metrics` of `transformers`
    Args:
        - p (tuple):      2 numpy arrays: predictions and true_labels
    Returns:
        - metrics (dict): f1 score on 
    """
    # (1)
    predictions, true_labels = p
    
    # (2)
    predicted_labels = np.where(predictions > 0, np.ones(predictions.shape), np.zeros(predictions.shape))
    metrics = {'f1_B-Skill':0,
               'f1_B-Knowledge':0,
               'f1_I-Skill':0,
               'f1_I-Knowledge':0,
               'macro_f1':0}
    
    # (3)
    cm = multilabel_confusion_matrix(true_labels.reshape(-1, n_labels), predicted_labels.reshape(-1, n_labels))
    
    # (4) 
    for label_idx, matrix in enumerate(cm):
        if label_idx == 4:
            continue # We don't care about the label "O"
        tp, fp, fn = matrix[1, 1], matrix[0, 1], matrix[1, 0]
        precision = divide(tp, tp + fp)
        recall = divide(tp, tp + fn)
        f1 = divide(2 * precision * recall, precision + recall)
        metrics[f"f1_{ids_to_labels[label_idx]}"] = f1
        
    # (5)
    macro_f1 = sum(list(metrics.values())) / (n_labels - 1)
    metrics["macro_f1"] = macro_f1
        
    return metrics
    
    
    
    
    
    
    


if __name__  == "__main__":
    from transformers import AutoTokenizer
    import numpy as np
    tokenizer = AutoTokenizer.from_pretrained("jjzha/jobbert-base-cased")
    
    example_text = "building a consent system for parents or shoppers ."
    mots = np.array(example_text.strip().split(" "))
    faux_label = np.ones(len(mots))

    
    toinp = tokenizer(example_text,return_tensors="pt")
    f_label = aligner_les_labels(toinp, faux_label)
    #print(f_label)
    #ds = SequencesData(df, tokenizer)
    #dl = t_data.DataLoader(ds,batch_size=2,shuffle = False)
    #for train_data, train_label in tqdm(dl):
     #   print(train_label)
      #  break