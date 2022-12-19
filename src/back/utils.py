import pandas as pd
import back.similarity as sim
import os
import random

#%%
def getKey(key) :
    return key + str(random.randint(0, 100000))

#%%
def getSkillList() :
    skillList = pd.read_csv('data/skill_list.csv', sep = '\t')
    return skillList

#%%
def writeJobCsv() :
    directory = 'data/jobs'
    jobs_data = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        sim.clean_document(filepath)
        jobs_data.append(sim.parsing_conll_document(filepath))
        jobs_data[-1].append(filename.split('.txt')[0])
        with open (filepath) as filetext: 
            url = filetext.readlines()[0].strip()
            jobs_data[-1].append(url) 

    #Each element of jobs_data is a list that represents one job and has [Skill, Knowledge, Job name, Url]
    table = {'Job': [jobs_data[0][2], jobs_data[1][2], jobs_data[2][2]], 
          'Skill': [str(jobs_data[0][0]).strip('[]'), str(jobs_data[1][0]).strip('[]'), str(jobs_data[2][0]).strip('[]')],
          'Knowledge': [str(jobs_data[0][1]).strip('[]'), str(jobs_data[1][1]).strip('[]'), str(jobs_data[2][1]).strip('[]')],
          'Url': [str(jobs_data[0][3]), str(jobs_data[1][3]), str(jobs_data[2][3])],
        }
    table_df = pd.DataFrame(data=table)

    text_file = open("data/jobs_db.csv", 'w')
    text_file.write(table_df.to_csv(index=False))
    text_file.close()

#%%
def writeUserCsv() :
    directory = 'data/users'
    users_data = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open (filepath) as filetext: 
            text = filetext.readlines()
            users_data.append([text[0].strip(), text[1].strip(), text[2].strip(), text[3].strip()]) 

    #Each element of users_data is a list that represents one user and has [Url, User name, Skill, Knowledge]
    table = {'User': [users_data[0][1], users_data[1][1], users_data[2][1], users_data[3][1], users_data[4][1], users_data[5][1], users_data[6][1], users_data[7][1], users_data[8][1], users_data[9][1]], 
          'Skill': [str(users_data[0][2]).strip('[]'), str(users_data[1][2]).strip('[]'), str(users_data[2][2]).strip('[]'), str(users_data[3][2]).strip('[]'), str(users_data[4][2]).strip('[]'), str(users_data[5][2]).strip('[]'), str(users_data[6][2]).strip('[]'), str(users_data[7][2]).strip('[]'), str(users_data[8][2]).strip('[]'), str(users_data[9][2]).strip('[]')],
          'Knowledge': [str(users_data[0][3]).strip('[]'), str(users_data[1][3]).strip('[]'), str(users_data[2][3]).strip('[]'), str(users_data[3][3]).strip('[]'), str(users_data[4][3]).strip('[]'), str(users_data[5][3]).strip('[]'), str(users_data[6][3]).strip('[]'), str(users_data[7][3]).strip('[]'), str(users_data[8][3]).strip('[]'), str(users_data[9][3]).strip('[]')],
          'Url': [str(users_data[0][0]), str(users_data[1][0]), str(users_data[2][0]), str(users_data[3][0]), str(users_data[4][0]), str(users_data[5][0]), str(users_data[6][0]), str(users_data[7][0]), str(users_data[8][0]), str(users_data[9][0])],
        }
    table_df = pd.DataFrame(data=table)

    text_file = open("data/users_db.csv", 'w')
    text_file.write(table_df.to_csv(index=False))
    text_file.close()

#%%
def longStringToList(longString) :
    temp_list = longString.split(sep=',')
    return longString.split(sep=',')

#%%
def getJobSimilarity(user) :
    #Return ['Job', 'Similarity', 'Url']
    #User = [Skill, Knowledge]
    jobs_data = pd.read_csv("data/jobs_db.csv")
    jobs_similarity = []
    for i in range(len(jobs_data)) :
        #jobs_similarity.append([jobs_data['Job'][i], sim.similarity([jobs_data['Skill'][i], jobs_data['Knowledge'][i]], user)])
        jobs_similarity.append(
            [jobs_data['Job'][i],
            sim.similarity([
                longStringToList(str(jobs_data['Skill'][i])), 
                longStringToList(str(jobs_data['Knowledge'][i]))
                ], user),
            jobs_data['Url'][i]])

    return pd.DataFrame(data=jobs_similarity).sort_values([1], ascending=False)

#%%
def getUsersSimilarity(job) :
    #Return ['User', 'Similarity', 'Url']
    #Job = [Skill, Knowledge]
    users_data = pd.read_csv("data/users_db.csv")
    users_similarity = []
    for i in range(len(users_data)) :
        users_similarity.append(
            [users_data['User'][i],
            sim.similarity([
                longStringToList(str(users_data['Skill'][i])), 
                longStringToList(str(users_data['Knowledge'][i]))
                ], job),
            users_data['Url'][i]])

    return pd.DataFrame(data=users_similarity).sort_values([1], ascending=False)

#%%
def getSkillKnowledgeRepartition(skillList, knowledgeList, src) :
    #Return the repartition of each skill and knowledge from skillList and knowledgeList in the database src
    #Return two dataframe : [Repartition for skills, Repartition for knowledges]
    #src == 'users' or 'job'
    skillRepartition = []
    knowledgeRepartition = []
    dict = {}

    if src == 'users':
        data = pd.read_csv("data/users_db.csv")
    else : 
        data = pd.read_csv("data/jobs_db.csv")
    
    for index in range(len(data)) :
        list = data['Skill'][index]
        for sk in list :
            if sk not in dict :
                dict.update({sk: 1})
            else :
                dict[sk] = dict[sk] + 1
    
    for skill in skillList :
        if skill in dict :
            skillRepartition.append(dict[skill])
        else : 
            skillRepartition.append(0)
    for knowledge in knowledgeList :
        if knowledge in dict :
            knowledgeRepartition.append(dict[knowledge])
        else :
            knowledgeRepartition.append(0)
    

    return pd.DataFrame(data=[skillList + knowledgeList, skillRepartition + knowledgeRepartition]).transpose().sort_values([1], ascending=False)

#%%

#writeJobCsv()
#writeUserCsv()
