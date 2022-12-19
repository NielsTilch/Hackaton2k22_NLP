import numpy as np
from sentence_transformers import SentenceTransformer,util
from torch import Tensor
import os

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


#%% 
def clean_document(document_path):
    temp = "data/jobs/test.txt"
    with open(document_path, "r") as input:
        with open(temp, "w") as output:
            for line in input:
                # if text matches then don't write it
                if "O	O" not in line and line.strip() != "":
                    output.write(line)
    os.replace(temp, document_path)


#%%
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

#%%
def parsing_text(text):
    skill=[]
    knowledge=[]

    i_skill=''
    i_knowledge=''
    
    for line in str(text).split('\n')[1:]:
        if len(line)>3:
            list=line.split('\t')

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
            elif list[2]!='I-Knowledge' and len(i_knowledge)!=0:
                knowledge.append(i_knowledge)
                i_knowledge=''

    return [skill, knowledge]

#%%
def similarity(job,candidate):
    # Compute embedding for both lists

    embedding_knowledge_candidate_list=[]
    embedding_knowledge_job_list = []
    embedding_skills_candidate_list=[]
    embedding_skills_job_list = []

    for element in job[0]:
        embedding_skills_job_list.append(model.encode(element, convert_to_tensor=True))

    for element in job[1]:
        embedding_knowledge_job_list.append(model.encode(element, convert_to_tensor=True))

    for element in candidate[0]:
        embedding_skills_candidate_list.append(model.encode(element, convert_to_tensor=True))

    for element in candidate[1]:
        embedding_knowledge_candidate_list.append(model.encode(element, convert_to_tensor=True))

    sub_score_skill_list=[]
    sub_score_knowledge_list = []

    for i,job_skill in enumerate(job[0]):
        for j,candidate_skill in enumerate(candidate[0]):
            if job_skill==candidate_skill:
                sub_score_skill_list.append(1)
                break

        max=Tensor([[0]])
        for candidate_skill in embedding_skills_candidate_list:
            test = util.pytorch_cos_sim(candidate_skill, embedding_skills_job_list[i])
            if test > max :
                max = test
        sub_score_skill_list.append(max.numpy()[0][0])



    for i,job_knowledge in enumerate(job[1]):
        for j,candidate_knowledge in enumerate(candidate[1]):
            if job_knowledge == candidate_knowledge:
                sub_score_knowledge_list.append(1)
                break

        max=Tensor([[0]])
        for candidate_knowledge in embedding_knowledge_candidate_list:
            test = util.pytorch_cos_sim(candidate_knowledge, embedding_knowledge_job_list[i])
            if test > max:
                max = test
        sub_score_knowledge_list.append(max.numpy()[0][0])

    var=(len(sub_score_knowledge_list)*np.mean(sub_score_knowledge_list) + len(sub_score_skill_list)*np.mean(sub_score_skill_list))/(len(sub_score_skill_list)+len(sub_score_knowledge_list))

    return var

