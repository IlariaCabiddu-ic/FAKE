import numpy as np

import model_trust
import warnings
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
import collections
import numpy as np
import math
import matplotlib.pyplot as plt

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

""" There are several source news, the assessment is for each of them """
def compute_expertise_a(df):
    a = np.arange(0.1, 1.1, 0.1)  # dominant parameter to the asymptotic value for the delta weight
    b = 1  # dominant parameter for the trend rate to the asymptotic parameter (when N_st is 100 we have an 80% of 0.3)
    Q_st = []
    dictionary_Q = {}
    exp = []
    d1 = []
    d2 = []
    d3 = []
    df["Message_based_converted"] = (df["Message_based"] + 1) / 2
    df = df.drop_duplicates()
    sources = df['Source'].unique()
    topics = list(sub_df['Topic'].unique())
    '''computation of M_st that is a percentage number of published news items 
    of a specific topic compared to the total number of news items of a source'''
    "Compute focus theme M_st"
    M_st = sub_df['Topic'].to_list()
    M_st = dict(collections.Counter(M_st))
    "Number of total news"
    N_s = sum(M_st.values())
    for i in M_st:
        M_st[i] = float(round(M_st[i] / N_s, 4))
    for value in a:
        "Compute technicality Q_st"
        for t in topics:
            df_sub = sub_df[sub_df['Topic'] == t]  # iteration for the each topic
            N_st = df_sub.shape[0]
            Q_st.append(sum(df_sub['Message_based_converted']) / (df_sub.shape[0]+1))  # divide by number of topic news P_st
            dictionary_Q[t] = sum(df_sub['Message_based_converted']) / (df_sub.shape[0]+1)
            delta1 = N_st/(abs((1/value)*N_st)+b)  # function1
            # delta2 = (value*N_st) / math.sqrt(b + N_st ** 2)  # function2
            # delta3 = value-value*math.exp(-b*N_st)  # funtion3
            d1.append(delta1)
            # d2.append(delta2)
            # d3.append(delta3)
            theta1 = 1 - delta1
            # theta2 = 1 - delta2
            # theta3 = 1 - delta3
            zipped_lists = zip(list(M_st.values()), Q_st)
            expertise1 = [delta1 * x + theta1*y for (x, y) in zipped_lists]
            expertise1 = [round(x, 2) for x in expertise1]
            # expertise2 = [delta2 * x + theta2 * y for (x, y) in zipped_lists]
            # expertise2 = [round(x, 2) for x in expertise2]
            # expertise3 = [delta3 * x + theta3 * y for (x, y) in zipped_lists]
            # expertise3 = [round(x, 2) for x in expertise3]
        exp.append(expertise1)

    return a,exp
def compute_expertise_b(df):
    a = 0.3  # dominant parameter to the asymptotic value for the delta weight
    b = np.array([1,10,100])  # dominant parameter for the trend rate to the asymptotic parameter (when N_st is 100 we have an 80% of 0.3)
    Q_st = []
    dictionary_Q = {}
    exp = []
    d1 = []
    d2 =[]
    d3 = []
    df["Message_based_converted"] = (df["Message_based"] + 1) / 2
    df = df.drop_duplicates()
    sources = df['Source'].unique()
    topics = list(sub_df['Topic'].unique())
    '''computation of M_st that is a percentage number of published news items 
    of a specific topic compared to the total number of news items of a source'''
    "Compute focus theme M_st"
    M_st = sub_df['Topic'].to_list()
    M_st = dict(collections.Counter(M_st))
    "Number of total news"
    N_s = sum(M_st.values())
    for i in M_st:
        M_st[i] = float(round(M_st[i] / N_s, 4))
    for value in b:
        "Compute technicality Q_st"
        for t in topics:
            df_sub = sub_df[sub_df['Topic'] == t]  # iteration for the each topic
            N_st = df_sub.shape[0]
            Q_st.append(sum(df_sub['Message_based_converted']) / (df_sub.shape[0]+1))  # divide by number of topic news P_st
            dictionary_Q[t] = sum(df_sub['Message_based_converted']) / (df_sub.shape[0]+1)
            # delta1 = N_st/(abs((1/a)*N_st)+value)  # function1
            # delta2 = (a*N_st) / math.sqrt(value + N_st ** 2)  # function2
            delta3 = a-a*math.exp(-value*N_st)  # funtion3
            # d1.append(delta1)
            # d2.append(delta2)
            d3.append(delta3)
            # theta1 = 1 - delta1
            # theta2 = 1 - delta2
            theta3 = 1 - delta3
            zipped_lists = zip(list(M_st.values()), Q_st)
            # expertise1 = [delta1 * x + theta1*y for (x, y) in zipped_lists]
            # expertise1 = [round(x, 2) for x in expertise1]
            # expertise2 = [delta2 * x + theta2 * y for (x, y) in zipped_lists]
            # expertise2 = [round(x, 2) for x in expertise2]
            expertise3 = [delta3 * x + theta3 * y for (x, y) in zipped_lists]
            expertise3 = [round(x, 2) for x in expertise3]
        exp.append(expertise3)

    return b,exp
dataframe = [
             ('T11', 'source1', 'Tech', 0.7, 0.8, '2022-01-31 00:00:00'),
             ('T12', 'source1', 'Tech', 0.7, 0.7, '2022-01-11 10:00:00'),
             ('T13', 'source1', 'Tech', 0.8, 0.8, '2021-12-30 14:30:40'),
             ('T14', 'source1', 'Tech', 0.7, 0.7, '2021-12-20 23:05:09'),
             ('T15', 'source1', 'Tech', 0.7, 0.8, '2021-11-06 13:44:44'),
             ('T16', 'source1', 'Tech', 0.9, 0.8, '2021-10-01 01:10:55'),
             ('T17', 'source1', 'Tech', 0.7, 0.7, '2021-09-18 01:10:55'),
             ('T18', 'source1', 'Tech', 0.7, 0.7, '2021-08-20 01:10:55'),
             ('T19', 'source1', 'Tech', 0.8, 0.8, '2021-07-13 01:10:55'),
             ('T110', 'source1', 'Tech', 0.8, 0.8, '2021-07-03 01:10:55'),
             ('T111', 'source1', 'Tech', 0.8, 0.7, '2021-06-01 00:00:00'),
             ('T112', 'source1', 'Tech', 0.9, 0.9, '2021-05-11 10:00:00'),
             ('T113', 'source1', 'Tech', 0.8, 0.7, '2021-03-30 14:30:40'),
             ('T114', 'source1', 'Tech', 0.8, 0.8, '2021-02-20 23:05:09'),
             ('T115', 'source1', 'Tech', 0.8, 0.8, '2021-01-06 13:44:44'),
             ('T116', 'source1', 'Tech', 0.7, 0.7, '2020-10-17 01:10:55'),
             ('T117', 'source1', 'Tech', 0.9, 0.9, '2020-09-18 01:10:55'),
             ('T118', 'source1', 'Tech', 0.9, 0.9, '2020-08-20 01:10:55'),
             ('T119', 'source1', 'Tech', 0.8, 0.8, '2020-07-13 01:10:55'),
             ('T120', 'source1', 'Tech', 0.7, 0.8, '2020-05-16 01:10:55'),
             ('T121', 'source1', 'Tech', 0.8, 0.8, '2020-01-31 00:00:00'),
             ('T122', 'source1', 'Tech', 0.9, 0.8, '2020-01-11 10:00:00'),
             ('T123', 'source1', 'Tech', 0.7, 0.8, '2019-12-30 14:30:40'),
             ('T124', 'source1', 'Tech', 0.9, 0.9, '2019-12-20 23:05:09'),
             ('T125', 'source1', 'Tech', 0.8, 0.9, '2019-11-06 13:44:44'),
             ('T126', 'source1', 'Tech', 0.9, 0.9, '2019-10-01 01:10:55'),
             ('T127', 'source1', 'Tech', 0.7, 0.7, '2019-09-18 01:10:55'),
             ('T128', 'source1', 'Tech', 0.8, 0.7, '2019-08-20 01:10:55'),
             ('T129', 'source1', 'Tech', 0.8, 0.8, '2019-07-13 01:10:55'),
             ('T130', 'source1', 'Tech', 0.9, 0.9, '2019-07-03 01:10:55'),
             ('T131', 'source1', 'Tech', 0.7, 0.7, '2019-06-01 00:00:00'),
             ('T132', 'source1', 'Tech', 0.9, 0.9, '2019-05-11 10:00:00'),
             ('T133', 'source1', 'Tech', 0.8, 0.8, '2019-03-30 14:30:40'),
             ('T134', 'source1', 'Tech', 0.7, 0.8, '2019-02-20 23:05:09'),
             ('T135', 'source1', 'Tech', 0.8, 0.7, '2019-01-06 13:44:44'),
             ('T136', 'source1', 'Tech', 0.9, 0.8, '2018-10-17 01:10:55'),
             ('T137', 'source1', 'Tech', 0.8, 0.8, '2018-09-18 01:10:55'),
             ('T138', 'source1', 'Tech', 0.9, 0.9, '2018-08-20 01:10:55'),
             ('T139', 'source1', 'Tech', 0.8, 0.8, '2018-07-13 01:10:55'),
             ('T140', 'source1', 'Tech', 0.8, 0.8, '2018-05-16 01:10:55'),
             # ----------------------another topic--------------------------
             ('A11', 'source1', 'Actuality', 0.8, 0.7, '2019-12-12 03:32:00'),
             # ('A12', 'source1', 'Actuality', 0.7, 0.7, '2019-11-01 11:00:10'),
             # ('A13', 'source1', 'Actuality', 0.9, 0.9, '2018-03-22 00:00:00'),
             # ('A16', 'source1', 'Actuality', 0.5, 0.7, '2017-11-12 23:32:00'),
             # ('A16', 'source1', 'Actuality', 0.8, 0.8, '2017-01-12 23:32:00'),

             # ----------------------ANOTHER SOURCE--------------------------
             ('T21', 'source2', 'Tech', 0.8, 0.8, '2022-01-31 00:00:00'),
             # ('T22', 'source2', 'Tech', 0.8, 0.7, '2022-01-11 10:00:00'),
             # ('T23', 'source2', 'Tech', 0.7, 0.7, '2021-12-30 14:30:40'),
             # ('T24', 'source2', 'Tech', 0.7, 0.8, '2021-12-20 23:05:09'),
             # ('T25', 'source2', 'Tech', 0.8, 0.8, '2021-11-06 13:44:44'),
             # ('T26', 'source2', 'Tech', 0.8, 0.8, '2021-10-01 01:10:55'),
             # ('T27', 'source2', 'Tech', 0.7, 0.7, '2021-09-18 01:10:55'),
             # ('T28', 'source2', 'Tech', 0.8, 0.7, '2021-08-20 01:10:55'),
             # ('T29', 'source2', 'Tech', 0.8, 0.8, '2021-07-13 01:10:55'),
             # ('T210', 'source2', 'Tech', 0.7, 0.8, '2021-07-03 01:10:55'),
             # # ----------------------another topic--------------------------
             # ('E21', 'source2', 'Economics', 0.4, 0.2, '2019-10-24 12:30:00'),
             # ('P21', 'source2', 'Politics', -0.2, 0.1, '2018-05-20 05:10:33'),
             # ('E22', 'source2', 'Economics', 0.2, 0.3, '2015-10-5 22:55:40'),
             ('A24', 'source2', 'Actuality', -0.8, -0.8, '2019-01-01 00:50:00')]


# Create a DataFrame object
df = pd.DataFrame(dataframe, columns=['ID', 'Source', 'Topic', 'Feedback', 'Message_based', 'Datetime'])
# print(df)
exp_new = []
"-----------------analysis of each present source--------------"
list_colours = ['orange','purple','orange','purple']
sources = list(df['Source'].unique())
topics = df['Topic'].unique()
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for s in sources:
    sub_df = df[df['Source'] == s]  # iteration for the each topic
    # a,expertise =compute_expertise_a(df=sub_df)
    b, expertise = compute_expertise_b(df=sub_df)
    # expertisem = model_trust.compute_expertise(df=sub_df,topics=sub_df['Topic'].unique())

    for i in range(len(df['Topic'].unique())):
        exp_new.append([item[i] for item in expertise])

    # for p in range(len(exp_new)):
    #     axs[p % 2].plot(a, exp_new[p], color=list_colours[math.floor(p/2)])
    #     axs[p % 2].set_title(('Expertise ', topics[p%2]))
    #     axs[p % 2].set_ylim([0, 1])
    #     axs[p % 2].set_yticks(np.arange(0, 1, step=0.1))
    #     axs[p % 2].set_xticks(np.arange(0, 1.1, step=0.1))
    for p in range(len(exp_new)):
        axs[p % 2].plot(b, exp_new[p], color=list_colours[math.floor(p/2)])
        axs[p % 2].set_title(('Expertise ', topics[p%2]))
        axs[p % 2].set_ylim([0, 1])
        axs[p % 2].set_yticks(np.arange(0, 1, step=0.1))


print(exp_new)
for ax in axs.flat:
    # ax.set(xlabel='a value', ylabel='Metric value')
    ax.set(xlabel='b value', ylabel='Metric value')
    leg = ax.legend(df['Source'].unique())
    for i, j in enumerate(leg.legendHandles):
        j.set_color(list_colours[i])
plt.show()

    # relevance = model_trust.compute_relevance(df=sub_df, topics=topics)
    # print("The relevance of analysed source '", s, "' is: ", relevance)
    # goodwill = model_trust.compute_goodwill(df=sub_df, topics=topics, relevance=relevance)
    # print("The goodwill of analysed source '", s, "' is: ", goodwill)
    # coherence = model_trust.compute_coherence(df=sub_df, topics=topics)
    # print("The coherence of analysed source '", s, "' is: ", coherence)
    # trust = model_trust.compute_trust(expertise=expertise, goodwill=goodwill,
    #                                   coherence=coherence, topics=topics)
    # print("The trust of analysed source '", s, "' is: ", trust)
