"""This file  is used to simulate OOA behaviour of the source.
In detail we simulate a periodic fake news pubblication with different period """
import pandas as pd
import numpy as np
import warnings
from pandas.core.common import SettingWithCopyWarning
import matplotlib.pyplot as plt
import model_trust

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def random_dates(start, end, n):  # generate random date for the dataset's datetimes
    start_u = start.value // 10 ** 9
    end_u = end.value // 10 ** 9
    return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s')


start = pd.to_datetime('2018-01-01')  # start date
end = pd.to_datetime('2022-02-01')  # end date

dataframe = pd.read_csv('../Dataset/TEST_01_with_topics.csv')
df = dataframe[["source", "label", "message-based", "mainTopic"]]
df = df.rename(columns={"source": "Source", "label": "label", "message-based": "Message_based", "mainTopic": "Topic"})
df["Feedback"] = df["Message_based"]  # copy of message based value
df['ID'] = pd.Series(range(0, df.shape[0]-1))
datetimes = pd.Series(random_dates(start, end, df.shape[0]))
datetimes = datetimes.sort_values(ascending=True)  # sorted datetimes insertion
datetimes = datetimes.reset_index(drop=True)
df['Datetime'] = datetimes
dfupdate = df[df['Source'] == 'nytimes']  # selection of one source and one topic
dfupdate= dfupdate.reset_index(drop=True)

df1 = dfupdate[dfupdate['Topic'] == 'Politics']  # 170 news
# df1 = dfupdate[dfupdate['Topic'] == 'Crime & security']  # 104 news
# df1 = dfupdate[dfupdate['Topic'] == 'Business & Economy']  # 80 news

df1 = df1.reset_index(drop=True)

E1 = []
G1 = []
C1 = []
T1 = []
step = [10, 20, 30, 40]
fig, axs = plt.subplots(4, 1, figsize=(12, 12))


for j in range(df1.shape[0]):
    sub_d = df1[0:j + 1]  # application of trust model for each native news entry
    sub_topics = list(sub_d['Topic'].unique())
    "application of trust model for each news entry"
    expertise1 = model_trust.compute_expertise(df=sub_d, topics=sub_topics)
    relevance1 = model_trust.compute_relevance(df=sub_d, topics=sub_topics)
    goodwill1 = model_trust.compute_goodwill(df=sub_d, topics=sub_topics, relevance=relevance1)
    coherence1 = model_trust.compute_coherence(df=sub_d, topics=sub_topics)
    trust1 = model_trust.compute_trust(expertise=expertise1, goodwill=goodwill1,
                                       coherence=coherence1, topics=sub_topics)

    E1.append(*expertise1.values())
    G1.append(*goodwill1.values())
    C1.append(*coherence1.values())
    T1.append(*trust1.values())
x1 = np.arange(0, len(E1))
axs[0].plot(x1, E1, '--')
axs[0].set_title('Expertise')
axs[0].set_ylim([0, 1])
axs[0].set_yticks(np.arange(0, 1, step=0.1))

axs[1].plot(x1, G1, '--')
axs[1].set_title('Goodwill')
axs[1].set_ylim([0, 1])
axs[1].set_yticks(np.arange(0, 1, step=0.1))

axs[2].plot(x1, C1, '--')
axs[2].set_title('Coherence behaviour')
axs[2].set_ylim([0, 1])
axs[2].set_yticks(np.arange(0, 1, step=0.1))

axs[3].plot(x1, T1, '--')
axs[3].set_title('Trust')
axs[3].set_ylim([0, 1])
axs[3].set_yticks(np.arange(0, 1, step=0.1))

"""----modification of dataset, simulation of OOA------"""
for s in step:
    df1new = df1.copy()
    for i in range(s,df1.shape[0], 2*s):
        df1update = df1.iloc[i:i+s].copy()
        df1update['Message_based'] = -df1update['Message_based']
        df1update['Feedback'] = -df1update['Feedback']
        df1update['label'] = -df1update['label']
        df1new.loc[df1update.index, :] = df1update[:]
    E = []
    G = []
    C = []
    T = []
    for j in range(df1new.shape[0]):
        sub_d = df1new[0:j + 1]
        sub_topics = list(sub_d['Topic'].unique())
        expertise = model_trust.compute_expertise(df=sub_d, topics=sub_topics)
        relevance = model_trust.compute_relevance(df=sub_d, topics=sub_topics)
        goodwill = model_trust.compute_goodwill(df=sub_d, topics=sub_topics, relevance=relevance)
        coherence = model_trust.compute_coherence(df=sub_d, topics=sub_topics)
        trust = model_trust.compute_trust(expertise=expertise, goodwill=goodwill,
                                          coherence=coherence, topics=sub_topics)

        E.append(*expertise.values())
        G.append(*goodwill.values())
        C.append(*coherence.values())
        T.append(*trust.values())

    x = np.arange(0, len(E))
    axs[0].plot(x, E)
    axs[0].set_title('Expertise')
    axs[0].set_ylim([0, 1])
    axs[0].set_yticks(np.arange(0, 1, step=0.1))
    axs[0].set_xticks(np.arange(0, 170, step=10))

    axs[1].plot(x, G)
    axs[1].set_title('Goodwill')
    axs[1].set_ylim([0, 1])
    axs[1].set_yticks(np.arange(0, 1, step=0.1))
    axs[1].set_xticks(np.arange(0, 170, step=10))

    axs[2].plot(x, C)
    axs[2].set_title('Coherence behaviour')
    axs[2].set_ylim([0, 1])
    axs[2].set_yticks(np.arange(0, 1, step=0.1))
    axs[2].set_xticks(np.arange(0, 170, step=10))

    axs[3].plot(x, T)
    axs[3].set_title('Trust')
    axs[3].set_ylim([0, 1])
    axs[3].set_yticks(np.arange(0, 1, step=0.1))
    axs[3].set_xticks(np.arange(0, 170, step=10))


for ax in axs.flat:
    ax.set(xlabel='News requests during the time', ylabel='Metric value')
    ax.legend(['dataset without modification', 'fake/real news alternation = 10', 'fake/real news alternation = 20',
               'fake/real news alternation = 30', 'fake/real news alternation = 40'],
              loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout(rect=[0, 0, 1, 1])
    plt.figure(figsize=(20, 20), dpi=80)
plt.show()
