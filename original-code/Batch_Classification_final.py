# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 20:07:52 2017

@author: Rodrigo Coronado, Nasim Taba, Pratheek Devaraj
# Will use persited models, clasify batch comments and output the data in 2 ways:
# 1: Matrix: Classification of 68 topics per comment. This is used to received all the results at once in a spreadsheet.
# 2: A structured table with 1 record per classified comment. This is used for reporting purposes.
"""

# Libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import multilayer_perceptron
from sklearn.externals import joblib
import pandas as pd
import numpy as np

#sys.path.insert(0, ".\\Script Bundle") 
survey_df = pd.read_csv("C:/Users/rorui/Desktop/data/combined.csv", encoding='ISO-8859-1')

topics=['Career_Personal_Development',
              'CPD_Improve_new_employee_orientation', 
              'CPD_Improve_performance',
              'CPD_Improve_training',
              'CPD_Provide_opportunities_advancement',
              'CPD_other', 
              'Compensation_Benefits',
              'CB_Ensure_salary_parity_across_gov',
              'CB_Ensure_salary_parity_with_other_orgs',
              'CB_Improve_medical',
              'CB_Increase_salary',
              'CB_Review_job_classifications',
              'CB_other',
              'Engagement_Workplace_Culture',
              'EWC_Act_on_engagement',
              'EWC_Address_discrimination',
              'EWC_Improve_morale',
              'EWC_Treat_employees_better',
              'EWC_Value_diversity',
              'Executives',
              'Exec_Hold_executives_accountable',
              'Exec_Improve_communication',
              'Exec_Improve_stability',
              'Exec_Make_yourselves_visible',
              'Exec_Strengthen_quality_of_executive_leaders',
              'Exec_other',
              'Flexible_Work_Environment',
              'FWE_Broaden_LWS_implementation',
              'FWE_Increase_flexibility_location',
              'FWE_Increase_flexibility_schedule',
              'FWE_other',
              'Hiring_Promotion',
              'HP_Ensure_hiring_and_promotions_merit_based',
              'HP_Focus_on_succession_planning',
              'HP_Review_job_descriptions',
              'HP_other',
              'Recognition_Empowerment',
              'RE_Enable_staff_to_make_decisions',
              'RE_Listen_to_staff_input',
              'RE_Make_better_use_of_skills',
              'RE_Provide_more_recognition',
              'Supervisors',
              'Sup_Cultivate_effective_teamwork',
              'Sup_Hold_employees_accountable',
              'Sup_Hold_managers_accountable',
              'Sup_Improve_communication_between_employees',
              'Sup_Strengthen_quality_of_supervisory_leadership',
              'Sup_other',
              'Stress_Workload',
              'SW_Hire_more_staff',
              'SW_Improve_productivity_and_efficiency',
              'SW_Review_workload_expectations',
              'SW_Support_a_healthy_workplace',
              'SW_other',
              'Tools_Equipment_Physical_Environment',
              'TEPE_Improve_facilities',
              'TEPE_Provide_better_equipment',
              'TEPE_Provide_better_furniture',
              'TEPE_Provide_better_hardware',
              'TEPE_Upgrade_improve_software',
              'TEPE_other',
              'Vision_Mission_Goals',
              'VMG_Improve_collaboration',
              'VMG_Improve_program',
              'VMG_Improve_transparency',
              'VMG_Pay_attention_to_the_public_interest',
              'VMG_Review_funding_or_budget',
              'VMG_other']

# Data Frames for output
dataframe1 = survey_df # input batch survey
dataframeR = pd.DataFrame() # output 1, formated like batch input
dataframef = pd.DataFrame(columns=('id','subcategory','prediction', 'category')) # output 2, formated in a structured way

# Classification routine. Using RNN (Neural Network)
dataframeR['ID.2'] = survey_df['ID.2'] # add ID column
for i in topics:
    # Hydrate Bag of Words. Predifined Bag of Words
    vectorizer = joblib.load('C:/Users/rorui/Documents/myGitHub/capstone_bc_stats_students/Deliverables/Model/Final/bow'+i+'.pkl')
    X = vectorizer.transform(dataframe1['AQ3345_13_CLEAN'].values.astype('U'))
        
    # Hydrate Pretrained model. Predifined Nerual Network
    model = joblib.load('C:/Users/rorui/Documents/myGitHub/capstone_bc_stats_students/Deliverables/Model/Final/'+i+'.pkl')
    y_train = model.predict(X)

    # Dataframe1 - matrix
    dataframeR[i] = y_train.tolist()

    
# Output 1: Matrix 68 topics
dataframeR.to_csv('resultant_matrix.csv')

# Output 2: Structured table with 1 row per classified comment. Output for reporting purposes
for i in topics:    
    df_temp = pd.DataFrame({'id':dataframeR['ID.2'],'subcategory':np.repeat(i,len(dataframe1)), 
                            'prediction':dataframeR[i], 'category':np.repeat(i.split('_')[0],len(dataframe1))})
    dataframef = dataframef.append(df_temp)

dataframef = dataframef[dataframef['prediction']==1]
dataframef.to_csv('resultant.csv') 
