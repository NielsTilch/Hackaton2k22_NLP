import torch
import numpy as np
import BertPack.Dataset_and_train_functions as dtf
from transformers import BertForTokenClassification, BertModel, BertForMaskedLM
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn.functional as F
import BertPack.readconllfile as rcf
from BertPack.MultiLabelsTokenClassification import BertForTokenMul


def fromClassesToknowledge_skill(classes,words):
    knowledge=[]
    skill=[]
    i_knowledge = ''
    i_skill = ''
    for i,element in enumerate(classes):

        if element == 0 or element == 1:
            i_skill=i_skill+' '+words[i]
        if i_skill != '' and element != 0 and element != 1:
            skill.append(i_skill)
            i_skill=''

        if element == 2 or element == 3:
            i_knowledge=i_knowledge+' '+words[i]
        if i_knowledge != '' and element != 2 and element != 3:
            knowledge.append(i_knowledge)
            i_knowledge=''

        if i==len(words)-1:
            if i_skill != '':
                skill.append(i_skill)
            if i_knowledge!='':
                knowledge.append(i_knowledge)

    return skill,knowledge

def askBERT(text):

    example_text = text
    output_dir = "./src/BertPack/models"
    Model = BertForTokenMul.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    #example_text = "building a consent system for parents or shoppers ."

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        Model = Model.cuda()
    Model.eval()
    toinp = tokenizer(example_text, return_tensors="pt").to(device)
    #print(toinp.word_ids())
    output = Model(**toinp)
    logits = output['logits'].detach().numpy()[0]
    # print("logitttttt:",logits)
    predictions = np.where(logits > 0, np.ones(logits.shape), np.zeros(logits.shape))
    mot = {"Skill": [], "Knowledge": []}
    preid = {"Skill": None, "Knowledge": None}
    pretoken = {"Skill": "", "Knowledge": ""}
    for i, x in enumerate(predictions):
        index = np.where(x > 0)
        #print(index[0])

        for label_idx in index[0]:
            if label_idx == 4:
                continue
            else:
                key = dtf.ids_to_labels[label_idx].strip().split("-")[1]
                token = tokenizer.convert_ids_to_tokens(toinp["input_ids"][0])[i]
                wordid = toinp.word_ids()[i]
                if wordid == preid[key] and wordid != None:
                    pretoken[key] += token[2:]
                else:
                    #print(pretoken[key])
                    if pretoken[key] != "" and "#" not in pretoken[key]:
                        mot[key].append(pretoken[key])
                    preid[key] = wordid
                    pretoken[key] = token
    if pretoken["Skill"] not in mot["Skill"]:
        mot["Skill"].append(pretoken["Skill"])
    if pretoken["Knowledge"] not in mot["Knowledge"]:
        mot["Knowledge"].append(pretoken["Knowledge"])
    skill = mot["Skill"]
    knowledge = mot["Knowledge"]


    return skill,knowledge




def parsing_conll_document(document_path):
    skill=[]
    knowledge=[]

    i_skill=''
    i_knowledge=''

    fd = open(document_path,'r')
    for line in fd.readlines()[1:]:
        if len(line)>3:
            list=line.split('\t')
            list[2]=list[2][:-1]
            if list[1] == 'B-Skill':
                if list[0][-1] == ' ':
                    i_skill=list[0]
                else:
                    i_skill=list[0]+' '
            elif list[1] == 'I-Skill':
                if list[0][-1] == ' ':
                    i_skill = i_skill+list[0]
                else:
                    i_skill = i_skill+list[0]+' '


            elif list[1]!='I-Skill' and len(i_skill)!=0:
                skill.append(i_skill)
                i_skill=''

            if list[2] == 'B-Knowledge':
                if list[0][-1] == ' ':
                    i_knowledge = list[0]
                else:
                    i_knowledge = list[0] + ' '
            elif list[2] == 'I-Knowledge':
                if list[0][-1] == ' ':
                    i_knowledge = i_knowledge+list[0]
                else:
                    i_knowledge = i_knowledge+list[0]+' '
                i_knowledge = i_knowledge+list[0]
            elif list[2]!='I-Knowledge' and len(i_knowledge)!=0:
                knowledge.append(i_knowledge)
                i_knowledge=''

    return [skill, knowledge]
