from PIL import Image

from io import StringIO
import pandas as pd
import streamlit as st
import back.utils as utils
import back.similarity as sim
import  streamlit_toggle as tog
import time
import numpy as np



skillList = []
knowledgeList = []

def build() :
    col1, col2, col3, col4 = st.columns([3,1,3,3])
    col2.image(Image.open("data/images/logo_transparent.png"), caption=" ", width=150)
    col3.header('')
    col3.title('Welcome to S.O.Hess')

