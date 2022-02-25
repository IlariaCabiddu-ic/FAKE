import model_trust
import warnings
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
import matplotlib.pyplot as plt
import numpy as np

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
""" There are several source news, the assessment is for each of them """
dataframe = [
            ('T120', 'source1', 'Tech', 0.7, 0.8, '2020-05-16 01:10:55'),
            ('T119', 'source1', 'Tech', 0.8, 0.8, '2020-07-13 01:10:55'),
            ('T118', 'source1', 'Tech', 0.9, 0.9, '2020-08-20 01:10:55'),
            ('T117', 'source1', 'Tech', 0.9, 0.9, '2020-09-18 01:10:55'),
            ('T116', 'source1', 'Tech', 0.7, 0.7, '2020-10-17 01:10:55'),
            ('T115', 'source1', 'Tech', 0.8, 0.8, '2021-01-06 13:44:44'),
            ('T114', 'source1', 'Tech', 0.8, 0.8, '2021-02-20 23:05:09'),
            ('T113', 'source1', 'Tech', 0.8, 0.7, '2021-03-30 14:30:40'),
            ('T112', 'source1', 'Tech', 0.9, 0.9, '2021-05-11 10:00:00'),
            ('T111', 'source1', 'Tech', 0.8, 0.7, '2021-06-01 00:00:00'),
            ('T110', 'source1', 'Tech', 0.8, 0.8, '2021-07-03 01:10:55'),
            ('T19', 'source1', 'Tech', 0.8, 0.8, '2021-07-13 01:10:55'),
            ('T18', 'source1', 'Tech', 0.7, 0.7, '2021-08-20 01:10:55'),
            ('T17', 'source1', 'Tech', -0.98,-0.98, '2021-09-18 01:10:55'),
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