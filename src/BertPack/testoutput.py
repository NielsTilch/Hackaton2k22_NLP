import torch
import numpy as np
from MultiLabelsTokenClassification import BertForTokenMul
from transformers import AutoTokenizer
import torch
from Dataset_and_train_functions import ids_to_labels
output_dir = "./models"
Model = BertForTokenMul.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)
example_text = "building a consent system for parents or shoppers ."


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    Model = Model.cuda()
Model.eval()
toinp = tokenizer(example_text,return_tensors="pt").to(device)
print(toinp.word_ids())
output = Model(**toinp)
logits = output['logits'].detach().numpy()[0]
#print("logitttttt:",logits)
predictions = np.where(logits > 0, np.ones(logits.shape), np.zeros(logits.shape))
x1 = predictions.reshape(-1,5)
mot = {"Skill":[],"Knowledge":[]}
preid = {"Skill":None,"Knowledge":None}
pretoken = {"Skill":None,"Knowledge":None}
for i,x in enumerate(predictions):
    index = np.where(x > 0)
    print(index[0])
    
    for label_idx in index[0] :
        if label_idx == 4:
            continue
        else:
            key = ids_to_labels[label_idx].strip().split("-")[1]
            token = tokenizer.convert_ids_to_tokens(toinp["input_ids"][0])[i]
            wordid = toinp.word_ids()[i]
            if wordid == preid[key]:
                pretoken[key] += token[2:]
            else :
                print(pretoken[key])
                if pretoken[key] != None:
                    mot[key].append(pretoken[key])
                preid[key] = wordid
                pretoken[key] = token
if pretoken["Skill"] not in mot["Skill"]:
    mot["Skill"].append(pretoken["Skill"])
if pretoken["Knowledge"] not in mot["Knowledge"]:
    mot["Knowledge"].append(pretoken["Knowledge"])
print(mot)
'''
restraite = [x for x in predictions if x >= 0]
indexneed = [i for i, x in enumerate(restraite) if x != 4]

mots = np.array(example_text.strip().split(" "))
sk = mots[indexneed]
print(sk)
'''