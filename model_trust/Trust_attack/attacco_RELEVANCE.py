import model_trust
import warnings
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
import matplotlib.pyplot as plt
import numpy as np

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
""" There are several source news, the assessment is for each of them """
dataframe = [
            ('T1120', 'source1', 'Tech', 0.8, 0.8, '2017-11-23 01:10:55'),
            ('T1119', 'source1', 'Tech', 0.8, 0.8, '2017-11-24 01:10:55'),
            ('T1118', 'source1', 'Tech', 0.9, 0.9, '2017-11-26 01:10:55'),
            ('T1117', 'source1', 'Tech', 0.8, 0.8, '2017-11-28 01:10:55'),
            ('T1116', 'source1', 'Tech', 0.8, 0.8, '2017-11-30 01:10:55'),
            ('T1115', 'source1', 'Tech', 0.7, 0.7, '2017-12-01 13:44:44'),
            ('T1114', 'source1', 'Tech', 0.7, 0.8, '2017-12-02 23:05:09'),
            ('T1113', 'source1', 'Tech', 0.8, 0.8, '2017-12-03 14:30:40'),
            ('T1112', 'source1', 'Tech', 0.9, 0.9, '2017-12-04 10:00:00'),
            ('T1111', 'source1', 'Tech', 0.7, 0.7, '2017-12-05 00:00:00'),
            ('T1110', 'source1', 'Tech', 0.9, 0.9, '2017-12-06 01:10:55'),
            ('T1109', 'source1', 'Tech', 0.8, 0.8, '2017-12-08 01:10:55'),
            ('T1108', 'source1', 'Tech', 0.8, 0.7, '2017-12-09 01:10:55'),
            ('T1107', 'source1', 'Tech', 0.7, 0.7, '2017-12-10 01:10:55'),
            ('T1106', 'source1', 'Tech', 0.9, 0.9, '2017-12-11 01:10:55'),
            ('T1105', 'source1', 'Tech', 0.8, 0.9, '2017-12-16 13:44:44'),
            ('T1104', 'source1', 'Tech', 0.9, 0.9, '2017-12-20 23:05:09'),
            ('T1103', 'source1', 'Tech', 0.7, 0.8, '2017-12-22 14:30:40'),
            #--------------same news but several request------------------
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:01'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:02'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:03'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:04'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:05'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:06'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:07'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:08'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:09'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:10'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:11'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:12'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:13'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:14'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:15'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:16'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:17'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:18'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:19'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:20'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:21'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:22'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:23'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:24'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:25'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:26'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:27'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:28'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:29'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:30'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:31'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:32'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:33'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:34'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:35'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:36'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:37'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:38'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:39'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:40'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2017-12-23 00:00:41'),

            ('T1101', 'source1', 'Tech', 0.8, 0.8, '2017-12-24 00:00:00'),
            ('T1100', 'source1', 'Tech', 0.7, 0.8, '2017-12-25 01:10:55'),
            ('T199', 'source1', 'Tech', 0.8, 0.8, '2017-12-26 01:10:55'),
            ('T198', 'source1', 'Tech', 0.9, 0.9, '2017-12-27 01:10:55'),
            ('T197', 'source1', 'Tech', 0.9, 0.9, '2017-12-28 01:10:55'),
            ('T196', 'source1', 'Tech', 0.7, 0.7, '2017-12-29 01:10:55'),
            ('T195', 'source1', 'Tech', 0.8, 0.8, '2018-01-01 13:44:44'),
            ('T194', 'source1', 'Tech', 0.7, 0.7, '2018-01-02 23:05:09'),
            ('T193', 'source1', 'Tech', 0.8, 0.7, '2018-01-03 14:30:40'),
            ('T192', 'source1', 'Tech', 0.9, 0.9, '2018-01-06 10:00:00'),
            ('T191', 'source1', 'Tech', 0.8, 0.7, '2018-01-07 00:00:00'),
            ('T190', 'source1', 'Tech', 0.8, 0.8, '2018-01-08 01:10:55'),
            ('T189', 'source1', 'Tech', 0.8, 0.8, '2018-01-09 01:10:55'),
            ('T188', 'source1', 'Tech', 0.7, 0.7, '2018-01-10 01:10:55'),
            ('T187', 'source1', 'Tech', 0.9, 0.8, '2018-01-14 01:10:55'),
            ('T186', 'source1', 'Tech', 0.9, 0.8, '2018-01-15 01:10:55'),
            ('T185', 'source1', 'Tech', 0.7, 0.8, '2018-01-16 13:44:44'),
            ('T184', 'source1', 'Tech', 0.7, 0.7, '2018-01-19 23:05:09'),
            ('T183', 'source1', 'Tech', 0.8, 0.8, '2018-01-20 14:30:40'),
            ('T182', 'source1', 'Tech', 0.7, 0.7, '2018-01-21 10:00:00'),
            ('T181', 'source1', 'Tech', 0.7, 0.8, '2018-01-27 00:00:00'),
            ('T180', 'source1', 'Tech', 0.8, 0.8, '2018-01-28 01:10:55'),
            ('T179', 'source1', 'Tech', 0.8, 0.8, '2018-01-29 01:10:55'),
            ('T178', 'source1', 'Tech', 0.9, 0.9, '2018-01-30 01:10:55'),
            ('T177', 'source1', 'Tech', 0.8, 0.8, '2018-02-01 01:10:55'),
            ('T176', 'source1', 'Tech', 0.9, 0.9, '2018-02-04 01:10:55'),
            ('T175', 'source1', 'Tech', 0.8, 0.7, '2018-02-06 13:44:44'),
            ('T174', 'source1', 'Tech', 0.7, 0.8, '2018-02-08 23:05:09'),
            ('T173', 'source1', 'Tech', 0.8, 0.8, '2018-02-10 14:30:40'),
            ('T172', 'source1', 'Tech', 0.9, 0.9, '2018-02-12 10:00:00'),
            ('T171', 'source1', 'Tech', 0.7, 0.7, '2018-02-14 00:00:00'),
            ('T170', 'source1', 'Tech', 0.9, 0.9, '2018-02-17 01:10:55'),
            ('T169', 'source1', 'Tech', 0.8, 0.8, '2018-02-18 01:10:55'),
            ('T168', 'source1', 'Tech', 0.8, 0.7, '2018-02-19 01:10:55'),
            ('T167', 'source1', 'Tech', 0.7, 0.7, '2018-02-20 01:10:55'),
            ('T166', 'source1', 'Tech', 0.9, 0.9, '2018-02-24 01:10:55'),
            ('T165', 'source1', 'Tech', 0.8, 0.9, '2018-02-26 13:44:44'),
            ('T164', 'source1', 'Tech', 0.9, 0.9, '2018-02-28 23:05:09'),
            ('T163', 'source1', 'Tech', 0.7, 0.8, '2018-03-01 14:30:40'),
            ('T162', 'source1', 'Tech', 0.9, 0.8, '2018-03-05 10:00:00'),
            ('T161', 'source1', 'Tech', 0.8, 0.8, '2018-03-07 00:00:00'),
            ('T160', 'source1', 'Tech', 0.7, 0.8, '2018-03-10 01:10:55'),
            ('T159', 'source1', 'Tech', 0.8, 0.8, '2018-03-12 01:10:55'),
            ('T158', 'source1', 'Tech', 0.9, 0.9, '2018-03-13 01:10:55'),
            ('T157', 'source1', 'Tech', 0.9, 0.9, '2018-03-16 01:10:55'),
            ('T156', 'source1', 'Tech', 0.7, 0.7, '2018-03-17 01:10:55'),
            ('T155', 'source1', 'Tech', 0.8, 0.8, '2018-03-19 13:44:44'),
            ('T154', 'source1', 'Tech', 0.7, 0.7, '2018-03-25 23:05:09'),
            ('T153', 'source1', 'Tech', 0.8, 0.7, '2018-03-30 14:30:40'),
            ('T152', 'source1', 'Tech', 0.9, 0.9, '2018-04-01 10:00:00'),
            ('T151', 'source1', 'Tech', 0.8, 0.7, '2018-04-05 00:00:00'),
            ('T150', 'source1', 'Tech', 0.8, 0.8, '2018-04-09 01:10:55'),
            ('T149', 'source1', 'Tech', 0.8, 0.8, '2018-04-13 01:10:55'),
            ('T148', 'source1', 'Tech', 0.7, 0.7, '2018-04-15 01:10:55'),
            ('T147', 'source1', 'Tech', 0.9, 0.8, '2018-04-18 01:10:55'),
            ('T146', 'source1', 'Tech', 0.9, 0.8, '2018-04-25 01:10:55'),
            ('T145', 'source1', 'Tech', 0.7, 0.8, '2018-04-27 13:44:44'),
            ('T144', 'source1', 'Tech', 0.7, 0.7, '2018-04-29 23:05:09'),
            ('T143', 'source1', 'Tech', 0.8, 0.8, '2018-05-05 14:30:40'),
            ('T142', 'source1', 'Tech', 0.7, 0.7, '2018-05-11 10:00:00'),
            ('T141', 'source1', 'Tech', 0.7, 0.8, '2018-05-15 00:00:00'),
            ('T140', 'source1', 'Tech', 0.8, 0.8, '2018-05-16 01:10:55'),
            ('T139', 'source1', 'Tech', 0.8, 0.8, '2018-07-13 01:10:55'),
            ('T138', 'source1', 'Tech', 0.9, 0.9, '2018-08-20 01:10:55'),
            ('T137', 'source1', 'Tech', 0.8, 0.8, '2018-09-18 01:10:55'),
            ('T136', 'source1', 'Tech', 0.8, 0.7, '2018-10-17 01:10:55'),

            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2018-12-23 00:00:41'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2018-12-23 00:00:42'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2018-12-23 00:00:43'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2018-12-23 00:00:44'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2018-12-23 00:00:45'),
            ('T1102', 'source1', 'Tech', 0.01, 0.01, '2018-12-23 00:00:46'),
            ('T135', 'source1', 'Tech', 0.8, 0.9, '2019-01-06 13:44:44'),
            ('T134', 'source1', 'Tech', 0.7, 0.8, '2019-02-20 23:05:09'),
            ('T133', 'source1', 'Tech', 0.8, 0.8, '2019-03-30 14:30:40'),
            ('T132', 'source1', 'Tech', 0.9, 0.9, '2019-05-11 10:00:00'),
            ('T131', 'source1', 'Tech', 0.7, 0.7, '2019-06-01 00:00:00'),
            ('T130', 'source1', 'Tech', 0.9, 0.9, '2019-07-03 01:10:55'),
            ('T129', 'source1', 'Tech', 0.8, 0.8, '2019-07-13 01:10:55'),
            ('T128', 'source1', 'Tech', 0.8, 0.7, '2019-08-20 01:10:55'),
            ('T127', 'source1', 'Tech', 0.8, 0.9, '2019-09-18 01:10:55'),
            ('T126', 'source1', 'Tech', 0.7, 0.8, '2019-10-01 01:10:55'),
            ('T125', 'source1', 'Tech', 0.8, 0.7, '2019-11-06 13:44:44'),
            ('T124', 'source1', 'Tech', 0.8, 0.8, '2019-12-20 23:05:09'),
            ('T123', 'source1', 'Tech', 0.8, 0.9, '2019-12-30 14:30:40'),
            ('T122', 'source1', 'Tech', 0.9, 0.8, '2020-01-11 10:00:00'),
            ('T121', 'source1', 'Tech', 0.8, 0.8, '2020-01-31 00:00:00'),
            ('T120', 'source1', 'Tech', 0.8, 0.7, '2020-05-16 01:10:55'),
            ('T119', 'source1', 'Tech', 0.8, 0.8, '2020-07-13 01:10:55'),
            ('T118', 'source1', 'Tech', 0.9, 0.9, '2020-08-20 01:10:55'),
            ('T117', 'source1', 'Tech', 0.9, 0.9, '2020-09-18 01:10:55'),
            ('T116', 'source1', 'Tech', 0.7, 0.7, '2020-10-17 01:10:55'),
            ('T115', 'source1', 'Tech', 0.8, 0.8, '2021-01-06 13:44:44'),
            ('T114', 'source1', 'Tech', 0.8, 0.8, '2021-02-20 23:05:09'),
            ('T113', 'source1', 'Tech', 0.7, 0.7, '2021-03-30 14:30:40'),
            ('T112', 'source1', 'Tech', 0.9, 0.9, '2021-05-11 10:00:00'),
            ('T111', 'source1', 'Tech', 0.8, 0.7, '2021-06-01 00:00:00'),
            ('T110', 'source1', 'Tech', 0.8, 0.8, '2021-07-03 01:10:55'),
            ('T19', 'source1', 'Tech', 0.8, 0.8, '2021-07-13 01:10:55'),
            ('T18', 'source1', 'Tech', 0.7, 0.7, '2021-08-20 01:10:55'),
            ('T17', 'source1', 'Tech', 0.9, 0.8, '2021-09-18 01:10:55'),
            ('T16', 'source1', 'Tech', 0.9, 0.8, '2021-10-01 01:10:55'),
            ('T15', 'source1', 'Tech', 0.7, 0.8, '2021-11-06 13:44:44'),
            ('T14', 'source1', 'Tech', 0.7, 0.7, '2021-12-20 23:05:09'),
            ('T13', 'source1', 'Tech', 0.8, 0.8, '2021-12-30 14:30:40'),
            ('T12', 'source1', 'Tech', 0.7, 0.7, '2022-01-11 10:00:00'),
            ('T11', 'source1', 'Tech', 0.7, 0.8, '2022-01-31 00:00:00') ]

df = pd.DataFrame(dataframe, columns=['ID', 'Source', 'Topic', 'Feedback', 'Message_based', 'Datetime'])
# print(df)
"-----------------analysis of each present source--------------"
sources = list(df['Source'].unique())
for s in sources:
    E = []
    G = []
    C = []
    T = []
    sub_df = df[df['Source'] == s]  # iteration for the each topic
    for j in range(df.shape[0]+1):
        sub_df =df[0:j+1]
        # print(sub_df)
        topics = list(sub_df['Topic'].unique())
        expertise = model_trust.compute_expertise(df=sub_df, topics=topics)
        # print("The expertise of analysed source '", s, "' is: ", expertise)

        relevance = model_trust.compute_relevance(df=sub_df, topics=topics)
        # print("The relevance of analysed source '", s, "' is: ", relevance)
        goodwill = model_trust.compute_goodwill(df=sub_df, topics=topics, relevance=relevance)
        # print("The goodwill of analysed source '", s, "' is: ", goodwill)
        coherence = model_trust.compute_coherence(df=sub_df, topics=topics)
        # print("The coherence of analysed source '", s, "' is: ", coherence)
        trust = model_trust.compute_trust(expertise=expertise, goodwill=goodwill,
                                          coherence=coherence, topics=topics)
        # print("The trust of analysed source '", s, "' is: ", trust)
        E.append(*expertise.values())
        G.append(*goodwill.values())
        C.append(*coherence.values())
        T.append(*trust.values())
# print(E)
# print(G)
# print(C)
# print(T)
fig, axs = plt.subplots(4, 2, figsize=(10,8))
x = np.arange(1, len(E)+1)
# print(x)
axs[0, 0].plot(x, E)
axs[0, 0].set_title('Expertise')
axs[0, 0].set_ylim([0.4,1])
axs[0, 0].set_yticks(np.arange(0.4, 1, step=0.1))

axs[0, 1].plot(x, E)
axs[0, 1].set_title('Expertise (zoom)')

axs[1, 0].plot(x, G, 'tab:orange')
axs[1, 0].set_title('Goodwill')
axs[1, 0].set_ylim([0.4,1])
axs[1, 0].set_yticks(np.arange(0.4, 1, step=0.1))

axs[1, 1].plot(x, G, 'tab:orange')
axs[1, 1].set_title('Goodwill (zoom)')

axs[2, 0].plot(x, C, 'tab:green')
axs[2, 0].set_title('Coherence behaviour')
axs[2, 0].set_ylim([0.4,1])
axs[2, 0].set_yticks(np.arange(0.4, 1, step=0.1))

axs[2, 1].plot(x, C, 'tab:green')
axs[2, 1].set_title('Coherence behaviour (zoom)')

axs[3, 0].plot(x, T, 'tab:red')
axs[3, 0].set_title('Trust')
axs[3, 0].set_ylim([0.4,1])
axs[3, 0].set_yticks(np.arange(0.4, 1, step=0.1))

axs[3, 1].plot(x, T, 'tab:red')
axs[3, 1].set_title('Trust (zoom)')



fig.tight_layout(pad=3)
plt.figure(figsize=(18, 16), dpi=80)
for ax in axs.flat:
    ax.set(xlabel='News requests during the time', ylabel='Metric value')
plt.show()