import streamlit as st
import os
import sys
import re
from io import StringIO
import numpy as np
import pandas as pd
import pickle
import sklearn
from scipy.stats import mode
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from sklearn.utils.validation import check_is_fitted, check_array
from missingpy import MissForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier as RC
from sklearn.utils.extmath import softmax



from streamlit import legacy_caching
legacy_caching.clear_cache()



# ======================== #
st.set_page_config(
    page_title="Predicting NCF Retention",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGE_STYLE = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """

st.markdown(PAGE_STYLE, unsafe_allow_html=True)
# ======================== #
 

def main():

    if 'button_pressed' not in st.session_state:
        st.session_state['button_pressed'] = False

    # Main panel
    st.title("Predicting NCF Retention")
    

    # ======================== #

    # Side panel

    st.sidebar.title("Select Timeframe")

    option = st.sidebar.selectbox(
     'After what term are you running predictions?',
     ('First term', 'Second term (first year)'))

    st.session_state['option'] = option

    st.sidebar.write('You selected:', option)

    st.sidebar.title("Data Upload")
    
    files_read_in = dict()

    # File uploaders

    # Retention
    retention_file = st.sidebar.file_uploader("Upload Retention file:", key=1)
    if retention_file:
         # Can be used wherever a "file-like" object is accepted:
         retention = load_data(retention_file)
         files_read_in['Retention'] = retention.columns

    # Course designations
    course_desig_file = st.sidebar.file_uploader("Upload Course designations file:", key=2)
    if course_desig_file:
         # Can be used wherever a "file-like" object is accepted:
         course_desig = load_data(course_desig_file)
         files_read_in['Course Designations'] = course_desig.columns

    # Course designations
    sat_file = st.sidebar.file_uploader("Upload SAT Scores file:", key=3)
    if sat_file:
         # Can be used wherever a "file-like" object is accepted:
         sat = load_data(sat_file)
         files_read_in['SAT'] = sat.columns

    # Course designations
    act_file = st.sidebar.file_uploader("Upload ACT Scores file:", key=4)
    if act_file:
         # Can be used wherever a "file-like" object is accepted:
         act = load_data(act_file)
         files_read_in['ACT'] = act.columns

    # Course designations
    gpa_file = st.sidebar.file_uploader("Upload High School GPA file:", key=5)
    if gpa_file:
         # Can be used wherever a "file-like" object is accepted:
         gpa = load_data(gpa_file)
         files_read_in['HS GPA'] = gpa.columns

    # Course designations
    col_gpa_file = st.sidebar.file_uploader("Upload College GPA file:", key=6)
    if col_gpa_file:
         # Can be used wherever a "file-like" object is accepted:
         col_gpa = load_data(col_gpa_file)
         files_read_in['College GPA'] = col_gpa.columns

    # Course designations
    scholarships_file = st.sidebar.file_uploader("Upload Scholarships file:", key=7)
    if scholarships_file:
         # Can be used wherever a "file-like" object is accepted:
         scholarships = load_data(scholarships_file)
         files_read_in['Scholarships'] = scholarships.columns

    # Course designations
    tests_file = st.sidebar.file_uploader("Upload AP/IB/AICE file:", key=8)
    if tests_file:
         # Can be used wherever a "file-like" object is accepted:
         tests = load_data(tests_file)
         files_read_in['AP/IB/AICE'] = tests.columns

    # Course designations
    rank_file = st.sidebar.file_uploader("Upload HS Rank file:", key=9)
    if rank_file:
         # Can be used wherever a "file-like" object is accepted:
         rank = load_data(rank_file)
         files_read_in['HS Rank'] = rank.columns

    # Course designations
    google_dist_file = st.sidebar.file_uploader("Upload Distances from NCF file:", key=10)
    if google_dist_file:
         # Can be used wherever a "file-like" object is accepted:
         google_dist = load_data(google_dist_file)
         files_read_in['Distances'] = google_dist.columns

    # Course designations
    zips_file = st.sidebar.file_uploader("Upload Zip Codes file:", key=11)
    if zips_file:
         # Can be used wherever a "file-like" object is accepted:
         zips = load_data(zips_file)
         files_read_in['Zip Codes'] = zips.columns

    # Course designations
    residency_file = st.sidebar.file_uploader("Upload Residency file:", key=12)
    if residency_file:
         # Can be used wherever a "file-like" object is accepted:
         residency = load_data(residency_file)
         files_read_in['Residency'] = residency.columns

    # Course designations
    income_file = st.sidebar.file_uploader("Upload Income file:", key=13)
    if income_file:
         # Can be used wherever a "file-like" object is accepted:
         income = load_data(income_file)
         files_read_in['Income'] = income.columns

    # Course designations
    parent_edu_file = st.sidebar.file_uploader("Upload Parent Education file:", key=14)
    if parent_edu_file:
         # Can be used wherever a "file-like" object is accepted:
         parent_edu = load_data(parent_edu_file)
         files_read_in['Parent Education'] = parent_edu.columns

    # SAP file upload depends on current time being run
    if st.session_state['option']=='Second term (first year)':
        sap_file = st.sidebar.file_uploader("Upload SAP file:", key=15)
        if sap_file:
            # Can be used wherever a "file-like" object is accepted:
            sap = load_data(sap_file)
            files_read_in['SAP'] = sap.columns



    # ========================== #




    # Dict of needed columns to check if user inputted all necessary columns

    cols_needed = dict()

    cols_needed['Retention'] = ['UNIV_ID', 'ADMIT_TERM', 'ADMIT_TYPE', 'BIRTH_DATE', 'GENDER_MASTER', 'RACE_MASTER']

    cols_needed['Course Designations'] = ['SQ_COUNT_STUDENT_ID', 'TERM', 'CLASS_TITLE', 'GRADABLE_INDICATOR', 'PART_TERM', 'CRS_SUBJ', 'CRS_NUMB', 'CRS_DIVS_DESC', 'ACAD_HIST_GRDE_DESC']

    cols_needed['SAT'] = ['N_NUMBER', 'TEST_REQ_CD', 'TEST_SCORE_TYP', 'TEST_SCORE_N']

    cols_needed['ACT'] = ['UNIV_ID', 'ACT_ENGLISH', 'ACT_MATH', 'ACT_READING', 'ACT_SCIENCE']

    cols_needed['HS GPA'] = ['UNIV_ID', 'GPA_HIGH_SCHOOL']

    cols_needed['College GPA'] = ['N_NUMBER', 'GPA_CODE', 'GPA']

    cols_needed['Scholarships'] = ['TermCode', 'SPRIDEN_ID', 'FundTitle', 'FORMATTED_PAID_AMT']

    cols_needed['AP/IB/AICE'] = ['N_NUMBER', 'TEST_DESC']

    cols_needed['HS Rank'] = ['N_NUMBER', 'HS_CLASS_RANK', 'HS_CLASS_SIZE']

    cols_needed['Distances'] = ['N_NUMBER', 'dist_from_ncf']

    cols_needed['Zip Codes'] = ['N_NUMBER', 'ZIP']

    cols_needed['Residency'] = ['N_NUMBER', 'TERM_ATTENDED', 'RESIDENCY']

    cols_needed['Income'] = ['SPRIDEN_ID', 'DEMO_TIME_FRAME', 'PARENTS_INCOME', 'STUDENT_INCOME', 'FAMILY_CONTRIB']

    cols_needed['Parent Education'] = ['SPRIDEN_ID', 'FatherHIGrade', 'MotherHIGrade']

    if st.session_state['option']=='Second term (first year)':
        cols_needed['Retention'].append('RETURNED_FOR_SPRING')
        cols_needed['SAP'] = ['N_NUMBER', 'TERM', 'SAPCODE']





    # ========================== #

    # Analysis button
    run_analysis = st.sidebar.button('Run analysis')

    if run_analysis:
        st.session_state.button_pressed = True

    # ========================= #

    # Check for missing columns from data upload

    missing_cols = False

    if st.session_state['button_pressed']:
        for k in files_read_in.keys(): # Iterate thru each dataset
            missing_col_list = []
            for col in cols_needed[k]: # Iterate thru each col in dataset
                if col not in files_read_in[k]: # Check if needed col not in file
                    missing_col_list.append(col) 
                    missing_cols = True
            if len(missing_col_list) > 0:
                st.markdown('#### Columns missing from '+str(k)+':')
                st.markdown(missing_col_list)
                st.markdown('Please add these columns to the respective dataset.')

    # ========================= #

    # Write the dataset upload schema if any file is not uploaded
    # or the "run analysis" button is not pressed
    if not st.session_state.button_pressed or not retention_file or not course_desig_file or not sat_file or not act_file or not gpa_file or not col_gpa_file or not scholarships_file or not tests_file or not rank_file or not google_dist_file or not zips_file or not residency_file or not income_file or not parent_edu_file:
        st.markdown("### Dataset Upload Schemas")
        st.markdown('''Please upload the following datasets, with at least the 
            specified columns (Note: Spelling, spacing, and capitalization is important).''')
        
        if st.session_state['option'] != 'Second term (first year)':
            table_schemas = open("Table_Schemas.txt", "r")
        # Change table schema appearance if option is full year
        elif st.session_state['option']=='Second term (first year)':
            table_schemas = open("Table_Schemas_fullyear.txt", "r")



        # THIS IS OLD CODE
        # # Change table schema appearance if option is full year
        # if st.session_state['option']=='Second term (first year)':
        #     legacy_caching.clear_cache()
        #     table_schemas = open("Table_Schemas_fullyear.txt", "r")
        # END OLD CODE


        st.markdown(table_schemas.read())








    # =============================================== #

    # Code to run after all files uploaded and user hit "Run Analysis" button


    if st.session_state['button_pressed'] and retention_file and course_desig_file and sat_file and act_file and gpa_file and col_gpa_file and scholarships_file and tests_file and rank_file and google_dist_file and zips_file and residency_file and income_file and parent_edu_file and missing_cols==False and st.session_state['option']=='First term':
        # Generate and store munged features
        # on which to run model
        retention = prepare_retention(retention, sat, act, col_gpa, gpa, tests, 
                        google_dist, residency, rank, zips, scholarships, course_desig,
                        income, parent_edu, sap=None)

        munged_df = prepare_first_term(retention)
        
        # Generate and store predictions
        prediction_df = output_preds(munged_df,
            cat_vars_path='static/grad_rate_pickles/Retention_first_term_cat_vars.pkl', 
            num_vars_path='static/grad_rate_pickles/Retention_first_term_num_vars.pkl', 
            stats_path='static/grad_rate_pickles/Retention_first_term_statistics.pkl', 
            scaler_path='static/grad_rate_pickles/Retention_first_term_scaler.pkl',
            model_path='static/grad_rate_pickles/Retention_first_term_model.pkl',
            model_type='ridge',
            cats = ['GENDER_M', 'IS_WHITE', 'IN_STATE', 'AP_IB_AICE_FLAG', 'CONTRACT_1_GRADE'])
 
        # Display predictions
        st.write('## Predictions')
        st.write(prediction_df)

        # Convert preds to csv and download
        pred_csv = prediction_df.to_csv(index=False).encode('utf-8')

        # Download button
        st.write('### Download Predictions')
        pred_download = st.download_button(
            "Download Predictions",
            pred_csv,
            "retention_preds.csv",
            "text/csv",
            key='download-course-csv'
            )
        



    elif st.session_state['button_pressed'] and retention_file and course_desig_file and sat_file and act_file and gpa_file and col_gpa_file and scholarships_file and tests_file and rank_file and google_dist_file and zips_file and residency_file and income_file and parent_edu_file and missing_cols==False and st.session_state['option']=='Second term (first year)':
        # Generate and store munged features
        # on which to run model
        retention = prepare_retention(retention, sat, act, col_gpa, gpa, tests, 
                        google_dist, residency, rank, zips, scholarships, course_desig,
                        income, parent_edu, sap)

        munged_df = prepare_full_year(retention)
        
        # Generate and store predictions
        prediction_df = output_preds(munged_df,
            cat_vars_path='static/grad_rate_pickles/Retention_full_year_cat_vars.pkl',
            num_vars_path='static/grad_rate_pickles/Retention_full_year_num_vars.pkl',
            stats_path='static/grad_rate_pickles/Retention_full_year_statistics.pkl',
            model_path='static/grad_rate_pickles/Retention_full_year_model.pkl',
            model_type='forest',
            cats = ['CONTRACT_1_GRADE', 
                     'CONTRACT_2_GRADE', 'SAP_GOOD', 'ISP_PASSED', 
                     'GENDER_M', 'IS_WHITE', 'IN_STATE', 'AP_IB_AICE_FLAG'])
        
        # Display predictions
        st.write('## Predictions')
        st.write(prediction_df)


        # Convert preds to csv and download
        pred_csv = prediction_df.to_csv(index=False).encode('utf-8')

        # Download button
        st.write('### Download Predictions')
        pred_download = st.download_button(
            "Download Predictions",
            pred_csv,
            "retention_preds.csv",
            "text/csv",
            key='download-course-csv'
            )



    return None












# =========================================================================================================== #

# FUNCTIONS
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_data(file_uploaded):
    if file_uploaded.name.rsplit('.', 1)[1] == 'csv':
        return pd.read_csv(file_uploaded, sep=',', encoding='utf-8')
    else:
        return pd.read_excel(file_uploaded)


def prepare_retention(retention, sat, act, col_gpa, gpa, tests, 
    google_dist, residency, rank, zips, scholarships, course_desig,
    income, parent_edu, sap=None):
    retention.ADMIT_TERM = retention.ADMIT_TERM.astype(int)

    # Filter out second baccalaureate students
    retention = retention.loc[retention['ADMIT_TYPE'] != "S"]

    # Initialize column of predicted second term for each student
    retention.ADMIT_TERM = retention.ADMIT_TERM.astype(str)
    retention['NEXT_TERM'] = np.where(retention.ADMIT_TERM.str.endswith("8"), retention.ADMIT_TERM.astype(int) + 93, np.nan)

    # Subset students after 200208
    retention.ADMIT_TERM = retention.ADMIT_TERM.astype(int)

    retention = retention.rename(columns={'UNIV_ID':'N_NUMBER'})


    # Merge SAT
    final_sat = prepare_sat(sat, act)
    retention = retention.merge(final_sat, how='left', on='N_NUMBER')

    # Merge GPA
    final_col_gpa = prepare_col_gpa(col_gpa)
    retention = retention.merge(final_col_gpa, how='left',on='N_NUMBER')

    final_hs_gpa = prepare_gpa(gpa)
    retention = retention.merge(final_hs_gpa, how='left',on='N_NUMBER')

    # Merge AP/IB/AICE tests
    taken_advanced = prepare_tests(tests)
    retention['AP_IB_AICE_FLAG'] = 0
    retention.loc[retention.N_NUMBER.isin(taken_advanced), 'AP_IB_AICE_FLAG'] = 1

    # Merge distances from NCF
    google_dist = prepare_google_dist(google_dist)
    retention = retention.merge(google_dist, how='left', on='N_NUMBER')

    # Merge In/Out-State residency
    residency = prepare_residency(residency)
    retention = retention.merge(residency, how='left', left_on=['N_NUMBER','ADMIT_TERM'], right_on=['N_NUMBER','TERM_ATTENDED']).drop(columns='TERM_ATTENDED')

    # Merge county data
    current_path = os.getcwd()

    county_zip_path = os.path.join(current_path, 'data/COUNTY_ZIP.csv')
    with open(county_zip_path, 'rb') as handle:
        county_zip = pd.read_csv(handle)

    education_path = os.path.join(current_path, 'data/Education.csv')
    with open(education_path, 'rb') as handle:
        education = pd.read_csv(handle)

    unemployment_path = os.path.join(current_path, 'data/Unemployment.xlsx')
    with open(unemployment_path, 'rb') as handle:
        unemployment = pd.read_excel(unemployment_path, sheet_name = "Unemployment Med HH Income", skiprows = 4)


    zips = prepare_zips(zips, county_zip, education, unemployment)
    retention = retention.merge(zips, how='left', on='N_NUMBER')

    # Merge HS rank
    final_rank = prepare_rank(rank)
    retention = retention.merge(final_rank, how='left', on='N_NUMBER')

    # Merge scholarships
    retention = prepare_scholarships(retention, scholarships)

    # Merge course designations
    retention = prepare_course_desig(retention, course_desig, term=st.session_state['option'])


    retention.GPA_HIGH_SCHOOL.fillna(retention.College_GPA, inplace=True)
    retention.loc[(retention.ADMIT_TYPE=='T') & ~retention.College_GPA.isna(), 'GPA_HIGH_SCHOOL'] = retention['College_GPA']
    retention.drop(columns='College_GPA', inplace=True)
    retention.rename(columns={'GPA_HIGH_SCHOOL':'GPA'}, inplace=True)
    retention = retention.dropna(subset=['GPA'])
    retention = retention.fillna({'TOTAL_FUNDS':0, 'UNSUB_FUNDS':0})
    retention['Admit_Age'] = (round(retention.ADMIT_TERM,-2) - round(retention.BIRTH_DATE,-4)/100)/100
    retention['SPRING_ADMIT'] = (retention.ADMIT_TERM.astype(str).str[-2:] == '01')
    retention.replace({'RACE_MASTER':{
        7:1, 6:0, 5:0, 4:0,
        3:0, 2:0, 1:0, 8:0,
        9:0
        }}, inplace=True)
    retention.rename(columns={'RACE_MASTER':'IS_WHITE'}, inplace=True)


    # Merge income
    income = prepare_income(income)
    retention = pd.merge(retention, income, left_on=['N_NUMBER', 'ADMIT_TERM'], right_on = ['SPRIDEN_ID','TERM'], how = 'left').drop(columns = ['SPRIDEN_ID','TERM'])

    # Merge parent education
    parent_edu = prepare_parent_edu(parent_edu)
    retention = retention.merge(parent_edu, how='left', left_on='N_NUMBER', right_on='SPRIDEN_ID').drop(columns='SPRIDEN_ID')

    if st.session_state['option'] == "Second term (first year)":
        # Merge SAP
        sap = prepare_sap(sap)
        retention = pd.merge(retention, sap[['TERM','N_NUMBER','SAP_GOOD']], how = 'left', left_on = ['N_NUMBER', 'NEXT_TERM'], right_on = ['N_NUMBER', 'TERM']).rename(columns={'SAPCODE':'SAP_NEXT_TERM'}).drop(columns='TERM')


    if st.session_state['option'] == 'Second term (first year)':
        retention = retention.rename(columns = {'RETURNED_FOR_SPRING' : 'FTIC_RETURNED_FOR_SPRING'})
        retention = retention[['N_NUMBER', 'GENDER_MASTER', 'IS_WHITE', 'ADMIT_TYPE',
                                'TEST_SCORE_N', 'SAT_MATH', 'GPA', 'IN_STATE', 'AP_IB_AICE_FLAG',
                                'dist_from_ncf', 'rank_percentile', 'TOTAL_FUNDS', 'UNSUB_FUNDS',
                                'FTIC_RETURNED_FOR_SPRING', 
                                'Percent of adults with a high school diploma only, 2015-19',
                                'Percent of adults with less than a high school diploma, 2015-19',
                                'COUNTY_UNEMPLOYMENT_RATE', 'ISP_PASSED', 'NUM_NONGRADABLE_TAKEN_1',
                                'NUM_NONGRADABLE_TAKEN_2', 'CONTRACT_1_GRADE', 'CONTRACT_2_GRADE',
                                'CREDITS_TAKEN_1', 'SAT_RATE_1', 'AVG_COURSE_LEVEL_1',
                                'DIVS_Humanities_1', 'DIVS_Natural_Science_1', 'DIVS_Social_Sciences_1',
                                'DIVS_Other_1', 'DIVS_Interdivisional_1', 'CREDITS_TAKEN_2',
                                'SAT_RATE_2', 'AVG_COURSE_LEVEL_2', 'DIVS_Humanities_2',
                                'DIVS_Natural_Science_2', 'DIVS_Social_Sciences_2', 'DIVS_Other_2',
                                'DIVS_Interdivisional_2', 'Admit_Age', 'SPRING_ADMIT', 'PARENTS_INCOME',
                                'STUDENT_INCOME', 'FAMILY_CONTRIB', 'FatherHIGrade',
                                'MotherHIGrade', 'SAP_GOOD']]

    elif st.session_state['option'] == 'First term':
        retention = retention[['N_NUMBER', 'GENDER_MASTER', 'IS_WHITE', 'ADMIT_TYPE',
                                'TEST_SCORE_N', 'SAT_MATH', 'GPA', 'IN_STATE', 'AP_IB_AICE_FLAG',
                                'dist_from_ncf', 'rank_percentile', 'TOTAL_FUNDS', 'UNSUB_FUNDS',
                                'Percent of adults with a high school diploma only, 2015-19',
                                'Percent of adults with less than a high school diploma, 2015-19',
                                'COUNTY_UNEMPLOYMENT_RATE', 'NUM_NONGRADABLE_TAKEN_1',
                                'CONTRACT_1_GRADE', 'CREDITS_TAKEN_1', 'SAT_RATE_1',
                                'AVG_COURSE_LEVEL_1', 'DIVS_Humanities_1', 'DIVS_Natural_Science_1',
                                'DIVS_Social_Sciences_1', 'DIVS_Other_1', 'DIVS_Interdivisional_1',
                                'Admit_Age', 'SPRING_ADMIT', 'PARENTS_INCOME', 'STUDENT_INCOME',
                                'FAMILY_CONTRIB', 'FatherHIGrade', 'MotherHIGrade']]


    return retention




def prepare_sat(sat, act):
    # Prepare SAT
    sat = sat.loc[sat.TEST_SCORE_TYP.isin(['M','RW']) & (sat.TEST_REQ_CD == 'S2')]

    # group by test and student ID and take max score of each subtest
    sat = sat[['N_NUMBER','TEST_REQ_CD','TEST_SCORE_TYP', 'TEST_SCORE_N']].groupby(['N_NUMBER','TEST_REQ_CD','TEST_SCORE_TYP'],as_index=False).agg({'TEST_SCORE_N':'max'})

    # Get just math scores for analysis
    sat_math = sat.loc[sat.TEST_SCORE_TYP=='M']

    sat_math = sat_math.rename(columns={'TEST_SCORE_N':'SAT_MATH'})[['N_NUMBER', 'SAT_MATH']]

    # Group by student ID and sum subscores
    sat = sat[['N_NUMBER','TEST_SCORE_N']].groupby('N_NUMBER', as_index=False).agg({'TEST_SCORE_N':'sum'})

    sat = sat.merge(sat_math, how='left', on='N_NUMBER')

    sat['TEST_SCORE_N'] = sat['TEST_SCORE_N']/2

    # Prepare ACT
    act = act[['UNIV_ID','ACT_ENGLISH','ACT_MATH','ACT_READING','ACT_SCIENCE']]

    act['ACT_Comp'] = round((act.ACT_ENGLISH + act.ACT_MATH + act.ACT_READING + act.ACT_SCIENCE)/4)

    act = act[['UNIV_ID','ACT_Comp', 'ACT_MATH']].rename(columns = {'UNIV_ID':'N_NUMBER'})

    vals = {
    36 : 1590, 35 : 1540, 34 : 1500,
    33 : 1460, 32 : 1430, 31 : 1400, 
    30 : 1370, 29 : 1340, 28 : 1310,
    27 : 1280, 26 : 1240, 25 : 1210,
    24 : 1180, 23 : 1140, 22 : 1110,
    21 : 1080, 20 : 1040, 19 : 1010,
    18 : 970, 17 : 930, 16 : 890,
    15 : 850, 14 : 800, 13 : 760,
    12 : 710, 11 : 670, 10 : 630,
    9 : 590
    }

    encodings = {'ACT_Comp': vals, 'ACT_MATH':vals}

    act = act.replace(encodings)

    act['ACT_Comp'] = act['ACT_Comp']/2
    act['ACT_MATH'] = act['ACT_MATH']/2

    act = act.rename(columns={'ACT_Comp':'TEST_SCORE_N',
                         'ACT_MATH':'SAT_MATH'})

    final_sat = pd.concat([sat, act]).groupby('N_NUMBER', as_index=False).agg({'TEST_SCORE_N':'max',
                                                                                 'SAT_MATH':'max'})

    return final_sat

def prepare_col_gpa(col_gpa):
    col_gpa = col_gpa[['N_NUMBER', 'COLLEGE_DATE', 'GPA_CODE', 'GPA']]
    fccol = col_gpa.loc[col_gpa.GPA_CODE == 'FCCOL']

    ccol = col_gpa.loc[col_gpa.GPA_CODE == 'CCOL']

    ccol = ccol.loc[~ccol.N_NUMBER.isin( list(set(fccol.N_NUMBER.values)) )]

    final_col_gpa = pd.concat([ccol, fccol], axis=0)

    final_col_gpa = final_col_gpa[['N_NUMBER','GPA']].rename(columns={'GPA':'College_GPA'})

    return final_col_gpa

def prepare_gpa(gpa):
    final_hs_gpa = gpa[['UNIV_ID','GPA_HIGH_SCHOOL']].rename(columns={'UNIV_ID':'N_NUMBER'})

    final_hs_gpa = final_hs_gpa.replace({0:np.nan, 9.8:np.nan})

    return final_hs_gpa

def prepare_tests(tests):
    tests = tests.loc[tests.TEST_DESC.str.startswith('AP') | tests.TEST_DESC.str.startswith('IB') | 
         tests.TEST_DESC.str.startswith('AICE')]

    taken_advanced = tests['N_NUMBER'].unique()

    return taken_advanced

def prepare_google_dist(google_dist):
    # Remove duplicate N Numbers
    google_dist = google_dist[['N_NUMBER', 'dist_from_ncf']].groupby('N_NUMBER', as_index=False).agg({'dist_from_ncf':'max'})

    return google_dist

def prepare_residency(residency):
    residency['IN_STATE'] = np.where(residency.RESIDENCY=='F',1,0)

    residency = residency[['N_NUMBER','TERM_ATTENDED','IN_STATE']]

    return residency

def prepare_zips(zips, county_zip, education, unemployment):
    zips['ZIP'] = zips['ZIP'].str.split('-').str[0]

    county_zip['zip'] = county_zip.zip.astype(str)

    zips = zips.merge(county_zip, how='left', left_on='ZIP', right_on='zip').drop(columns='zip')

    zips = zips.merge(education, how='left', left_on='county', right_on='FIPS Code')

    zips = zips[['N_NUMBER','ZIP',
       'Percent of adults with a high school diploma only, 2015-19',
       'Percent of adults with less than a high school diploma, 2015-19']]
    zips.loc[zips.duplicated(keep=False)].sort_values(by='N_NUMBER')

    zips = zips.groupby(['N_NUMBER', 'ZIP'], as_index=False).agg('mean')

    unemployment = unemployment[['FIPS_Code', 'Unemployment_rate_2019']]

    unemployment.rename(columns = {"Unemployment_rate_2019": "COUNTY_UNEMPLOYMENT_RATE"}, inplace = True)


    # Convert FIPS to ZIP
    unemployment = pd.merge(unemployment, county_zip, how = "left", left_on="FIPS_Code", right_on = "county")
    unemployment = unemployment[['zip', 'COUNTY_UNEMPLOYMENT_RATE']]

    unemployment = unemployment.dropna(subset=['zip'])

    unemployment = unemployment.groupby(['zip'], as_index=False).agg("mean")

    zips = pd.merge(zips, unemployment, how = 'left', left_on = ['ZIP'], right_on = ['zip']).drop(columns = ['zip'])

    zips = zips.groupby('N_NUMBER', as_index=False).agg('mean')

    return zips

def prepare_rank(rank):
    rank['rank_percentile'] = 1- (rank.HS_CLASS_RANK / rank.HS_CLASS_SIZE)

    rank = rank[['N_NUMBER', 'rank_percentile']].dropna()

    rank_dropped_dup = rank.drop_duplicates(subset=['N_NUMBER','rank_percentile'])

    final_rank = rank_dropped_dup.groupby('N_NUMBER', as_index=False).agg({'rank_percentile':'max'})

    return final_rank

def prepare_scholarships(retention, scholarships):
    scholarships = scholarships.rename(columns={'TermCode':'TERM', 'SPRIDEN_ID':'STUDENT_ID'})

    # Replace NA funds with zero
    scholarships.FORMATTED_PAID_AMT.fillna(0, inplace=True)

    # String match to extract unsubsizided funds
    unsub = scholarships.loc[scholarships.FundTitle.str.contains("Unsub", case = False)]

    # String match to extract subsidized funds
    scholarships = scholarships.loc[~scholarships.FundTitle.str.contains("Unsub", case = False)]

    # GroupBy ID/TERM to add up total funds awarded to each student, for each term
    scholarships = scholarships.groupby(["STUDENT_ID"])['FORMATTED_PAID_AMT'].agg(sum).reset_index(name='TOTAL_FUNDS')
    unsub = unsub.groupby(["STUDENT_ID"])['FORMATTED_PAID_AMT'].agg(sum).reset_index(name='TOTAL_FUNDS')

    # Subset records with non-zero funds
    scholarships = scholarships[scholarships['TOTAL_FUNDS'] > 0]
    unsub = unsub[unsub['TOTAL_FUNDS'] > 0]

    unsub.rename(columns={'TOTAL_FUNDS':'UNSUB_FUNDS'}, inplace=True)

    retention = pd.merge(retention, scholarships, left_on = ["N_NUMBER"], right_on = ["STUDENT_ID"], how = "left").drop(columns = ["STUDENT_ID"])
    retention = pd.merge(retention, unsub, left_on = ["N_NUMBER"], right_on = ["STUDENT_ID"], how = "left").drop(columns = ["STUDENT_ID"])

    return retention

def prepare_course_desig(retention, course_desig, term):
    course_desig.rename(columns = {'SQ_COUNT_STUDENT_ID' : 'ID'}, inplace=True)

    isps = course_desig.copy().loc[(course_desig.CRS_SUBJ == "ISP") & (course_desig.GRADABLE_INDICATOR == "Y")].reset_index(drop=True)

    # Replace TERMS ending in "1" with "2" (eg. 201801 -> 201802)
    course_desig.TERM = course_desig.TERM.astype(str)
    course_desig.TERM = course_desig.TERM.apply(lambda x: (x[:-1]+"1") if x.endswith("2") else x)
    course_desig.TERM = course_desig.TERM.astype(int)

    # Separate contracts out
    contract_desig = course_desig.loc[course_desig['CLASS_TITLE'].str.contains('Contract ')].rename(columns = {'SQ_COUNT_STUDENT_ID' : 'ID', 'ACAD_HIST_GRDE_DESC':'CONTRACT_GRADE'})

    make_unsat = ['Incomplete', 'Unsatisfactory (Preemptive)', 'Incomplete (Evaluation in Progress)']
    contract_desig.loc[contract_desig.CONTRACT_GRADE.isin(make_unsat),'CONTRACT_GRADE'] = "Unsatisfactory"


    contract_desig.dropna(subset=['CONTRACT_GRADE'], inplace=True)
    contract_desig.drop_duplicates(subset = ['ID', 'TERM'], inplace=True)

    # Count non-gradable courses taken by each student/term AND ADD TO CONTRACT_DESIG DATAFRAME
    contract_desig = course_desig.loc[course_desig.GRADABLE_INDICATOR == "N"].groupby(["ID", "TERM"]).GRADABLE_INDICATOR.size().reset_index().rename(columns = {"GRADABLE_INDICATOR":"NUM_NONGRADABLE_TAKEN"}).merge(contract_desig, how = "right", left_on=['ID','TERM'], right_on=['ID','TERM'])

    # replace na NUM_NONGRADABLE_TAKEN with zero
    contract_desig.NUM_NONGRADABLE_TAKEN.fillna(0, inplace = True)

    # Subset relevant columns
    contract_desig = contract_desig[['ID','TERM','NUM_NONGRADABLE_TAKEN','CONTRACT_GRADE']]

    # FILTER OUT WITHDRAWN STUDENTS
    contract_desig = contract_desig.loc[contract_desig.CONTRACT_GRADE != "Withdrawn"]

    retention = pd.merge(retention, contract_desig, how = "left", left_on = ['N_NUMBER','ADMIT_TERM'], right_on = ['ID','TERM']).drop(columns = ['ID','TERM'])

    if term == 'Second term (first year)':
        retention= pd.merge(retention, contract_desig, how = "left", left_on = ['N_NUMBER','NEXT_TERM'], right_on = ['ID','TERM']).drop(columns = ['ID','TERM'])

        retention.rename(columns = {'CONTRACT_GRADE_x':'CONTRACT_1_GRADE','CONTRACT_GRADE_y':'CONTRACT_2_GRADE',
                            'NUM_NONGRADABLE_TAKEN_x':'NUM_NONGRADABLE_TAKEN_1',
                            'NUM_NONGRADABLE_TAKEN_y':'NUM_NONGRADABLE_TAKEN_2'}, inplace = True)
    else:
        retention.rename(columns = {'CONTRACT_GRADE':'CONTRACT_1_GRADE',
                            'NUM_NONGRADABLE_TAKEN':'NUM_NONGRADABLE_TAKEN_1'}, inplace = True)

    # Course desig
    # Subset to gradable courses
    course_desig = course_desig.loc[(course_desig.CRS_SUBJ != "NCF") & (course_desig.CRS_SUBJ != "ISP") & (course_desig.GRADABLE_INDICATOR=='Y')]

    # Subset needed columns
    course_desig = course_desig[['ID', 'TERM', 'PART_TERM', 'CRS_NUMB', 'CRS_DIVS_DESC',
                  'ACAD_HIST_GRDE_DESC']]

    # Replace course credits with floats
    course_desig = course_desig.replace({'PART_TERM':{'1':1,
                                  'M1':0.5,
                                  'M2':0.5,
                                  '1MC':0.5}})

    # Extract course_level from CRS_NUMB
    course_desig['COURSE_LEVEL'] = [int(str(x)[0]) for x in course_desig.CRS_NUMB.tolist()]
    # Group course_level 5 & 6 values in with 4
    course_desig['COURSE_LEVEL'] = np.where(course_desig['COURSE_LEVEL'].gt(4), 4, course_desig['COURSE_LEVEL'])

    course_desig = course_desig.drop(columns='CRS_NUMB')

    # Values to mark as unsat
    make_unsat = ['Incomplete', 'Unsatisfactory (Preemptive)', 'Incomplete (Evaluation in Progress)']
    course_desig.loc[course_desig.ACAD_HIST_GRDE_DESC.isin(make_unsat),'ACAD_HIST_GRDE_DESC'] = "Unsatisfactory"

    # FILTER OUT RECORDS NOT IN SAT/UNSAT
    course_desig = course_desig.loc[course_desig.ACAD_HIST_GRDE_DESC.isin(['Satisfactory', 'Unsatisfactory'])]

    top_n = ['Humanities', 'Natural Science', 'Social Sciences', 'Other', 'Interdivisional']

    course_desig['CRS_DIVS_DESC'] = np.where(course_desig['CRS_DIVS_DESC'].isin(top_n), course_desig['CRS_DIVS_DESC'], "Other")

    # Encode CRS_DIVS_DESC as one-hot variables
    for n in top_n:
        
        dummy_colname = n.replace(" ", "_")
        dummy_colname = "DIVS_" + dummy_colname
        course_desig[dummy_colname] = np.where(course_desig['CRS_DIVS_DESC'] == n, 1, 0)

    course_desig = course_desig.drop(columns='CRS_DIVS_DESC')

    course_desig = course_desig.replace({'ACAD_HIST_GRDE_DESC':{'Satisfactory':1, 'Unsatisfactory':0}})

    course_desig = course_desig.groupby(['ID','TERM'],as_index=False).agg({'PART_TERM':'sum',
                                         'ACAD_HIST_GRDE_DESC':'mean',
                                         'COURSE_LEVEL':'mean',
                                         'DIVS_Humanities':'sum',
                                         'DIVS_Natural_Science':'sum',
                                         'DIVS_Social_Sciences':'sum',
                                         'DIVS_Other':'sum',
                                         'DIVS_Interdivisional':'sum'})
                                                                      

    course_desig = course_desig.rename(columns={'ACAD_HIST_GRDE_DESC':'SAT_RATE',
                                               'COURSE_LEVEL':'AVG_COURSE_LEVEL',
                                               'PART_TERM':'CREDITS_TAKEN'})

    retention = retention.merge(course_desig, how='left', left_on=['N_NUMBER','ADMIT_TERM'], right_on=['ID','TERM']).drop(columns=['ID','TERM'])
    
    if term == 'Second term (first year)':
        retention = retention.merge(course_desig, how='left', left_on=['N_NUMBER','NEXT_TERM'], right_on=['ID','TERM'], suffixes = ["_1", "_2"]).drop(columns=['ID','TERM'])
    else:
        suffixed_cols = retention[['CREDITS_TAKEN', 'SAT_RATE', 'AVG_COURSE_LEVEL', 'DIVS_Humanities', 'DIVS_Natural_Science', 'DIVS_Social_Sciences', 'DIVS_Other', 'DIVS_Interdivisional']].add_suffix("_1")
        retention = retention.drop(columns = ['CREDITS_TAKEN', 'SAT_RATE', 'AVG_COURSE_LEVEL', 'DIVS_Humanities', 'DIVS_Natural_Science', 'DIVS_Social_Sciences', 'DIVS_Other', 'DIVS_Interdivisional'])
        retention = pd.concat([retention, suffixed_cols], axis=1)

    retention = retention.replace({'CONTRACT_1_GRADE':{'Satisfactory':1, 'Unsatisfactory':0}, 'CONTRACT_2_GRADE':{'Satisfactory':1, 'Unsatisfactory':0}})

    
    # Only do this ISP munging when app is ran for the full year
    if term == 'Second term (first year)':
        # ISP munging
        # Relabel Unsatisfactory outcomes
        make_unsat = ['Incomplete', 'Unsatisfactory (Preemptive)', 'Incomplete (Evaluation in Progress)']
        isps.loc[isps.ACAD_HIST_GRDE_DESC.isin(make_unsat),'ACAD_HIST_GRDE_DESC'] = "Unsatisfactory"

        # FILTER OUT RECORDS NOT IN SAT/UNSAT
        isps = isps.loc[isps.ACAD_HIST_GRDE_DESC.isin(['Satisfactory', 'Unsatisfactory'])]

        isps = isps[['ID', 'TERM', 'ACAD_HIST_GRDE_DESC']]
        isps.rename(columns = {"ACAD_HIST_GRDE_DESC":"ISP_PASSED"}, inplace = True)
        isps.ISP_PASSED = np.where(isps.ISP_PASSED == "Satisfactory", 1, 0)

        spr_admit_ids = retention.loc[~retention.ADMIT_TERM.astype(str).str.endswith('08'), 'N_NUMBER'].tolist()
        
        isps.TERM = np.where((isps.TERM%10 != 8) & (~isps.ID.isin(spr_admit_ids)), isps.TERM-94, isps.TERM)

        isps = isps.groupby(['ID', 'TERM'], as_index=False).agg('max').sort_values(by=['ID','TERM'])
        isps = isps.drop_duplicates(subset='ID', keep='first').reset_index(drop=True)

        retention = pd.merge(retention, isps, how = 'left', left_on = ['N_NUMBER','ADMIT_TERM'], right_on = ['ID','TERM']).drop(columns = ['ID','TERM'])

    return retention

def prepare_income(income):
    income['DEMO_TIME_FRAME'] = income.DEMO_TIME_FRAME.apply(lambda x: str(x)[:4] + "08").astype("int64")
    income = income.rename(columns={"DEMO_TIME_FRAME": "TERM"})
    income.dropna(subset=['PARENTS_INCOME','STUDENT_INCOME','FAMILY_CONTRIB'], inplace=True)
    income = income[['SPRIDEN_ID', 'TERM','PARENTS_INCOME','STUDENT_INCOME','FAMILY_CONTRIB']]

    return income

def prepare_parent_edu(parent_edu):
    parent_edu['FatherHIGrade'] = parent_edu['FatherHIGrade'].astype(str).str.extract('(\d)').astype(float)
    parent_edu['MotherHIGrade'] = parent_edu['MotherHIGrade'].astype(str).str.extract('(\d)').astype(float)

    parent_edu = parent_edu.replace({'FatherHIGrade':{3:4, 4:3},
                    'MotherHIGrade':{3:4, 4:3}})

    parent_edu = parent_edu.drop(columns='First_Gen_flag')

    return parent_edu

def prepare_sap(sap):
    sap['TERM'] = sap.TERM.astype(str)

    replacements = {'GOOD**':'GOOD', 'WARN2':'WARN'}
    sap['TERM'] = sap.TERM.apply(lambda x: (x[:-1]+"1") if x.endswith("2") else x)

    sap['TERM'] = sap.TERM.astype(int)

    sap['SAP_GOOD'] = np.where(sap.SAPCODE=='GOOD', 1, 0)

    sap = sap[['TERM','N_NUMBER','SAP_GOOD']]

    return sap




# ============================================ #






def prepare_first_term(retention):

    # Drop Collinear Predictors
    retention = retention.drop(columns = ['DIVS_Social_Sciences_1'])

    # Drop Spring Admits
    retention = retention.loc[(~retention.SPRING_ADMIT)].reset_index(drop=True)

    # Replace 9.8 GPA with NA
    retention.replace({'GPA':{9.8:np.nan}}, inplace=True)

    retention = retention.replace({'FTIC_RETURNED_NEXT_FALL':{1:0, 0:1}}).rename(columns={'FTIC_RETURNED_NEXT_FALL':'FTIC_NO_FALL_RETURN'})

    retention = retention.loc[(ftic_df.ADMIT_TYPE == 'F')]

    retention = retention.fillna({'UNSUB_FUNDS':0})

    retention = retention.dropna(subset=['SAT_RATE_1', 'CONTRACT_1_GRADE'])

    retention.drop(columns=['failed_to_grad', 'SPRING_ADMIT', 'ADMIT_TYPE', 'NEXT_TERM', 
                        'FTIC_RETURNED_FOR_SPRING', 'ISP_PASSED','SAP_GOOD'], inplace=True)

    retention = retention.replace({'failed_to_grad':{True:1, False:0},
                                'GENDER_MASTER':{'M':1,'F':0}}
                                )
    retention.rename(columns={'ADMIT_TYPE':'IS_TRANSFER', 'GENDER_MASTER':'GENDER_M'}, inplace=True)

    # =================================== #
    # Treat Outliers
    treatoutliers(retention, columns=['GPA', 'dist_from_ncf', 'TOTAL_FUNDS', 
                                    'Percent of adults with a high school diploma only, 2015-19',
                                    'Percent of adults with less than a high school diploma, 2015-19',
                                    'COUNTY_UNEMPLOYMENT_RATE', 'PARENTS_INCOME', 'STUDENT_INCOME',
                                    'FAMILY_CONTRIB'], factor = 3)



    return retention





def prepare_full_year(retention):


    # Drop FTIC students who returned for Spring but did not have TERM 2 data
    retention = retention.loc[~((retention.FTIC_RETURNED_FOR_SPRING==1) & retention.SAT_RATE_2.isna())]

    # Drop Collinear Features
    retention = retention.drop(columns = ['FAMILY_INCOME', 'DIVS_Natural_Science_2',
                                        'DIVS_Natural_Science_1','FTIC_RETURNED_FOR_SPRING',
                                        'SAT_RATE_2'])

    # Drop Spring Admits
    retention = retention.loc[retention.SPRING_ADMIT==0]

    # Recode response
    retention = retention.replace({'FTIC_RETURNED_NEXT_FALL':{1:0, 0:1}}).rename(columns={'FTIC_RETURNED_NEXT_FALL':'FTIC_NO_FALL_RETURN'})

    # Subset incoming freshmen
    retention = retention.loc[(ftic_df.ADMIT_TYPE == 'F')]

    # Replace NA unsubsidized funds with zero
    retention = retention.fillna({'UNSUB_FUNDS':0})

    retention = retention.dropna(subset=['SAT_RATE_1', 'CONTRACT_1_GRADE'])

    retention.drop(columns=['failed_to_grad', 'SPRING_ADMIT', 'ADMIT_TYPE'], inplace=True)

    retention = retention.replace({'failed_to_grad':{True:1, False:0},
                                'GENDER_MASTER':{'M':1,'F':0}}
                                )
    retention.rename(columns={'ADMIT_TYPE':'IS_TRANSFER', 'GENDER_MASTER':'GENDER_M'}, inplace=True)

    retention = retention.fillna({'CREDITS_TAKEN_2':0,
                            'SAT_RATE_2':0,
                            'AVG_COURSE_LEVEL_2':0,
                            'DIVS_Humanities_2':0,
                            'DIVS_Natural_Science_2':0,
                            'DIVS_Social_Sciences_2':0,
                            'DIVS_Other_2':0,
                            'DIVS_Interdivisional_2':0,
                            'CONTRACT_2_GRADE':0,
                            'NUM_NONGRADABLE_TAKEN_2':0})

    retention = retention.drop(columns='NEXT_TERM')

    # =================================== #
    # Treat Outliers
    treatoutliers(retention, columns=['GPA', 'dist_from_ncf', 'TOTAL_FUNDS', 
                                    'Percent of adults with a high school diploma only, 2015-19',
                                    'Percent of adults with less than a high school diploma, 2015-19',
                                    'COUNTY_UNEMPLOYMENT_RATE', 'PARENTS_INCOME', 'STUDENT_INCOME',
                                    'FAMILY_CONTRIB'], factor=3)

    return retention



def output_preds(munged_df, cat_vars_path, num_vars_path, stats_path, model_path, cats, model_type, scaler_path=None):

    # Take IDs for prediction output
    predictions = munged_df[['N_NUMBER']]
    # DF to run model on
    munged_df = munged_df.drop(columns='N_NUMBER')

    # ================================ #
    # Read in pickled imputers
    current_path = os.getcwd()

    imputer = MissForest(criterion=("mse","gini"), oob_score=True, random_state=22, verbose=0)

    num_vars_path = os.path.join(current_path, num_vars_path)
    with open(num_vars_path, 'rb') as handle:
        num_vars = pickle.load(handle)

    cat_vars_path = os.path.join(current_path, cat_vars_path)
    with open(cat_vars_path, 'rb') as handle:
        cat_vars = pickle.load(handle)

    stats_path = os.path.join(current_path, stats_path)
    with open(stats_path, 'rb') as handle:
        statistics = pickle.load(handle)

    imputer.num_vars_ = num_vars
    imputer.cat_vars_ = cat_vars
    imputer.statistics_ = statistics


    # Imputing
    test_imputed = imputer.transform(munged_df)
    munged_df = pd.DataFrame(data = test_imputed,
        columns = munged_df.columns)


    # ================================ #
    # Scaling numerical features
    # (only if model is ridge regression)

    x_num = munged_df.drop(columns=cats)
    x_cat = munged_df[cats]

    num_cols = x_num.columns
    
    if scaler_path!=None:
        # Read in pickled scaler
        scaler_path = os.path.join(current_path, scaler_path)
        with open(scaler_path, 'rb') as handle:
            scl = pickle.load(handle)

        x_num = pd.DataFrame(scl.transform(x_num), columns=num_cols)

        munged_df = pd.concat([x_num, x_cat], axis=1)

    # ================================ #

    # Read in pickeled models
    model_path = os.path.join(current_path, model_path)
    with open(model_path, 'rb') as handle:
        model = pickle.load(handle)

    # Predicting
    if model_type=='ridge':
        d = model.decision_function(munged_df)
        d_2d = np.c_[-d, d]
        preds = softmax(d_2d)
    elif model_type=='forest':
        preds = model.predict_proba(munged_df)

    # Take prob of leaving
    preds = [item[1] for item in preds]

    # Add to prediction df
    predictions['Prob of NOT Grad. on Time'] = preds

    predictions = predictions.sort_values(by='Prob of NOT Grad. on Time', ascending=False)

    return  predictions



def treatoutliers(df, columns=None, factor=3, method='IQR', treament='cap'):
    """
    Removes the rows from self.df whose value does not lies in the specified standard deviation
    :param columns:
    :param in_stddev:
    :return:
    """
#     if not columns:
#         columns = self.mandatory_cols_ + self.optional_cols_ + [self.target_col]
    if not columns:
        columns = df.columns
    
    for column in columns:
        if method == 'STD':
            permissable_std = factor * df[column].std()
            col_mean = df[column].mean()
            floor, ceil = col_mean - permissable_std, col_mean + permissable_std
        elif method == 'IQR':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            floor, ceil = Q1 - factor * IQR, Q3 + factor * IQR
        
        if treament == 'remove':
            df = df[(df[column] >= floor) & (df[column] <= ceil)]
        elif treament == 'cap':
            df[column] = df[column].clip(floor, ceil)
            
    return None






if __name__ == "__main__":
    main()

