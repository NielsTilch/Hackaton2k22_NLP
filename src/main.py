from PIL import Image

from streamlit_option_menu import option_menu
import streamlit as st
import view.jobSeeker as jobSeeker
import view.employer as employer
import view.home as home

st.set_page_config(page_title='S.O.Hess', 
    page_icon=Image.open('data/images/logo.png'),
    layout="wide")

with st.sidebar:
    tabs = option_menu(None, ['Home', 'Job Seeker', 'Employer'], 
        icons=['house-door', 'search', 'graph-up'], 
        menu_icon="cast", default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "purple"},
        }
    )

if tabs =='Home':
    home.build()

elif tabs == 'Job Seeker':
    jobSeeker.build()

elif tabs == 'Employer':
    employer.build()
