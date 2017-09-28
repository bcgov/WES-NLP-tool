# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:09:05 2017

@author: Pratheek Devaraj, Nasim Taba, Rodrigo Coronado
# Script to persit Bag of Words and Neural Network classification models
"""

# LIBRARIES
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, zero_one_loss
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import multilayer_perceptron
from sklearn.externals import joblib
import pandas as pd
import numpy as np

# Paths
data_path = "C:/Users/rorui/Desktop/data/all_data2.csv"
bow_path = "C:/Users/rorui/Desktop/data/Final/"
rnn_path = "C:/Users/rorui/Desktop/data/Final/"

# DATA: Partition data Set
survey_df = pd.read_csv(data_path, encoding='ISO-8859-1')
mask = np.random.rand(len(survey_df)) < 0.7
train_df = survey_df[mask]
test_df = survey_df[~mask]

# TOPICS
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

# RESULT ARRAYS: accuracy, precision, recall arrays
train_accuracy=[]
test_accuracy=[]
train_precision=[]
test_precision=[]
train_recall=[]
test_recall=[]

# VECTORIZER NLP: used to create bag of words
vectorizer = CountVectorizer(analyzer = 'word', tokenizer = None, preprocessor = None, stop_words= 'english', max_features = 200)

# ROUTINE: Presist bag of words and RNN models.
for i in topics:
    # BAG OF WORDS: Create and Persist bag of words
    cpd_df = survey_df[survey_df[i]==1] # take records from topic only
    vectorizer.fit(cpd_df['AQ3345_13_CLEAN'].values.astype('U'))
    joblib.dump(vectorizer, bow_path + 'bow'+ i +'.pkl', protocol=2)
    
    # VECTORIZE COMMENTS
    X = vectorizer.transform(train_df['AQ3345_13_CLEAN'].values.astype('U'))
    Xtest = vectorizer.transform(test_df['AQ3345_13_CLEAN'].values.astype('U'))
    y = train_df[i]
    ytest = test_df[i]
    
    # RNN MODEL: Create and Persist model
    RNN=MLPClassifier(alpha=0.01, hidden_layer_sizes=(50,50)).fit(X, y)
    joblib.dump(RNN, rnn_path + i +'.pkl', protocol=2)
    
    # PREDICT
    y_trainl1= RNN.predict(X)
    y_validl1 = RNN.predict(Xtest)
    
    # METRICS. Accuracy, Precision, Recall
    train_accuracy.append(1- (zero_one_loss(y, y_trainl1)))
    test_accuracy.append(1-(zero_one_loss(ytest, y_validl1)))
    train_precision.append(precision_score(y,y_trainl1))
    test_precision.append(precision_score(ytest, y_validl1))
    train_recall.append(recall_score(y,y_trainl1))
    test_recall.append(recall_score(ytest, y_validl1))

    
# ***************************************
# TESTING ROUTINE: Using persited models
# ***************************************
dataframeR = pd.DataFrame() # Resultant datafram
dataframe1 = survey_df # Input batch comments
for i in topics:
    # Hydrate Bag of Words
    vectorizer = joblib.load('C:/Users/rorui/Documents/myGitHub/capstone_bc_stats_students/Deliverables/Model/Final/bow'+i+'.pkl')
    X = vectorizer.transform(dataframe1['AQ3345_13_CLEAN'].values.astype('U'))
        
    # Hydrate Pretrained model
    model = joblib.load('C:/Users/rorui/Documents/myGitHub/capstone_bc_stats_students/Deliverables/Model/Final/'+i+'.pkl')
    y_train = model.predict(X)

    # Output: Matrix 68 topics
    dataframeR[i] = y_train.tolist()