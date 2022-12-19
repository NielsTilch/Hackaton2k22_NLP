from io import StringIO
import pandas as pd
import streamlit as st
import back.utils as utils
import back.similarity as sim
import  streamlit_toggle as tog
import time
import numpy as np
import BertPack.askBert as bert

skillList = []
knowledgeList = []
mainSkillList = []
mainKnowledgeList = []

def build() :
    st.title("Welcome Employer")
    
    with st.expander("Add manually :"):
        col1, col2 = st.columns(2)

        addSkill = col1.selectbox('', utils.getSkillList(), index=1)
        if addSkill == "add skill": 
            col1_1, col2_1 = st.columns(2)
            inputSkill = col1_1.text_input('type your skill')

        col2.subheader(" ")
        
        validateSkill = col2.button("Validate")
        if validateSkill : 
            if addSkill == "add skill":
                toAdd = inputSkill
                inputSkill = ""
            else :
                toAdd = addSkill
            if toAdd not in skillList :
                skillList.append(toAdd)
                st.success("Skill added")
            else :
                st.warning("Skill already in list")
        
        
    with st.expander("Import from file :"):
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            text = uploaded_file.getvalue().decode("utf-8")
            #st.write(StringIO(text).read())
            st.text_area(label="sb", value=StringIO(text).read(), height=700, label_visibility="collapsed")
            #res = sim.parsing_text(text)
            res = bert.askBERT(text)
            if len(np.intersect1d(skillList,res[0]))!=len(res[0]):
                skillList.extend(set(res[0]))
            if len(np.intersect1d(knowledgeList,res[1]))!=len(res[1]):
                knowledgeList.extend(set(res[1]))


    with st.expander("Skills and Knowledges :"):

        toggleDisplay = tog.st_toggle_switch(label=" ", 
            key="Key1", 
            default_value=False, 
            label_after = False, 
            inactive_color = '#D3D3D3', 
            active_color="#F28500", 
            track_color="#880ED4"
        ) 

        st.header('Skill List :')
        if toggleDisplay == True:
            st.write(skillList)
        else : 
            for skill in set(skillList) :
                colA, colB = st.columns([1,30])
                with colA :
                    checkBox = st.checkbox(" ", key=skill + "s_toMain")
                    if checkBox and skill not in mainSkillList :
                        mainSkillList.append(skill)
                    elif not checkBox and skill in mainSkillList : 
                        mainSkillList.remove(skill)
                with colB :    
                    if st.button(label=skill, key=skill + "_s") :
                        skillList.remove(skill)
                
        st.write('---')
        st.header('Knowledge List :')
        if toggleDisplay == True:
            st.write(knowledgeList)
        else :
            for knowledge in set(knowledgeList) :
                colA, colB = st.columns([1, 30])
                with colA :
                    checkBox = st.checkbox(" ", key=knowledge + "k_toMain")
                    if checkBox and knowledge not in mainKnowledgeList :
                        mainKnowledgeList.append(knowledge)
                    elif not checkBox and knowledge in mainKnowledgeList : 
                        mainKnowledgeList.remove(knowledge)
                with colB :    
                    if st.button(label=knowledge, key=knowledge + "_k") :
                        knowledgeList.remove(knowledge)
        
        if mainSkillList != [] :
            st.write('---')
            st.header('Main skill List :')
            if toggleDisplay == True:
                st.write(mainSkillList)
            else : 
                for skill in set(mainSkillList) :
                    if st.button(label=skill, key=skill + 's_inMain') :
                        mainSkillList.remove(skill)
        
        if mainKnowledgeList != [] :
            st.write('---')
            st.header('Main knowledge List :')
            if toggleDisplay == True:
                st.write(mainKnowledgeList)
            else :
                for knowledge in set(mainKnowledgeList) :
                    if st.button(label=knowledge, key=knowledge + 'k_inMain') :
                        mainKnowledgeList.remove(knowledge)


    with st.expander("See similarities :"):
        if len(skillList) > 0 or len(knowledgeList) > 0 or uploaded_file is not None :
            if st.button(label="Clear all") :
                skillList.clear()
                knowledgeList.clear()
                st.experimental_rerun()

        if len(skillList) > 0 or len(knowledgeList) > 0 :
            buttonCalculateSimilarities = st.button(label="Calculate similarities") 
            buttonCalculateRepartition = st.button(label="Calculate repartition") 
            
            if len(mainSkillList) > 0 or len(mainKnowledgeList) > 0 :
                buttonCalculateMainSimilarities = st.button(label="Calculate main similarities") 
                buttonCalculateMainRepartition = st.button(label="Calculate main repartition") 
            
            with st.spinner("Loading..."):
                if buttonCalculateSimilarities:
                    st.dataframe(utils.getUsersSimilarity([skillList, knowledgeList]))
                    st.success("Similarities successfully calculated")
                if buttonCalculateRepartition:
                    st.dataframe(utils.getSkillKnowledgeRepartition(skillList, knowledgeList, "users"))
                    st.success("Repartitions successfully calculated")
                
                if len(mainSkillList) > 0 or len(mainKnowledgeList) > 0 :
                    if buttonCalculateMainSimilarities:
                        st.dataframe(utils.getUsersSimilarity([mainSkillList, mainKnowledgeList]))
                        st.success("Main similarities successfully calculated")
                    if buttonCalculateMainRepartition:
                        st.dataframe(utils.getSkillKnowledgeRepartition(mainSkillList, mainKnowledgeList, "users"))
                        st.success("Main repartitions successfully calculated")
        else :
            st.warning("Please, tell us more about you")

