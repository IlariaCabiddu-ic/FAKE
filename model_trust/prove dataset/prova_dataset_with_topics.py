import pandas as pd
import numpy as np
import warnings
from pandas.core.common import SettingWithCopyWarning
import matplotlib.pyplot as plt
from itertools import chain
import model_trust
import utils

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def random_dates(start, end, n):
    start_u = start.value // 10 ** 9
    end_u = end.value // 10 ** 9
    return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s')


start = pd.to_datetime('2018-01-01')
end = pd.to_datetime('2022-02-01')

dataframe = pd.read_csv('../Dataset/TEST_01_with_topics.csv')
df = dataframe[["source", "label", "message-based","mainTopic"]]
df = df.rename(columns={"source":"Source", "label":"label", "message-based":"Message_based", "mainTopic":"Topic"})
# df = df.loc[0:199]
# print(d)
df["Feedback"] = df["Message_based"]
# topics = pd.Series(['Economics','Politics', 'Tech', 'Actuality'])
# topics= topics.repeat((df.shape[0]/topics.shape[0])+1)
# topics = topics.reset_index(drop=True)
# topics = topics[0:df.shape[0]]
# df['Topic'] = topics[0:df.shape[0]]
# df = df.sample(frac=1)
df['ID'] = pd.Series(range(0,df.shape[0]-1))
datetimes = pd.Series(random_dates(start, end, df.shape[0]))
df['Datetime'] = datetimes
df = df.sort_values(by="Datetime")
df = df.reset_index(drop=True)

"-----------------analysis of each present source--------------"

sources = list(df['Source'].unique())
for s in sources:
    print("source ",s)
    sub_et = []
    sub_gt = []
    sub_ct = []
    sub_tt = []
    sub_e = []
    sub_g = []
    sub_c = []
    sub_t = []
    E = []
    G = []
    C = []
    T = []
    sub_df = df[df['Source'] == s]  # iteration for the each source
    for j in range(sub_df.shape[0]):
        sub_d = sub_df[0:j+1]
        # print(j)
        sub_topics = list(sub_d['Topic'].unique())
        expertise = model_trust.compute_expertise(df=sub_d, topics=sub_topics)
        exp = list(expertise.values())
        # print("The expertise of analysed source '", s, "' is: ", expertise)

        relevance = model_trust.compute_relevance(df=sub_d, topics=sub_topics)
        # print("The relevance of analysed source '", s, "' is: ", relevance)
        goodwill = model_trust.compute_goodwill(df=sub_d, topics=sub_topics, relevance=relevance)
        good = list(goodwill.values())
        # print("The goodwill of analysed source '", s, "' is: ", goodwill)
        coherence = model_trust.compute_coherence(df=sub_d, topics=sub_topics)
        coh = list(coherence.values())
        # print("The coherence of analysed source '", s, "' is: ", coherence)
        trust = model_trust.compute_trust(expertise=expertise, goodwill=goodwill,
                                          coherence=coherence, topics=sub_topics)
        tru = list(trust.values())
        E.append(exp)
        G.append(good)
        C.append(coh)
        T.append(tru)
    sub_et = utils.reorder_metrics(df=df, metric=E)
    sub_gt = utils.reorder_metrics(df=df, metric=G)
    sub_ct = utils.reorder_metrics(df=df, metric=C)
    sub_tt = utils.reorder_metrics(df=df, metric=T)

    for k in range(len(df['Topic'].unique())):
        sub_e.append(list(chain(sub_exp[k] for sub_exp in sub_et)))
        sub_g.append(list(chain(sub_good[k] for sub_good in G)))
        sub_c.append(list(chain(sub_coh[k] for sub_coh in C)))
        sub_t.append(list(chain(sub_tru[k] for sub_tru in T)))
    # fig, axs = plt.subplots(4, 2, figsize=(10, 10))
    # x = np.arange(0, len(sub_et))
    # for e in sub_e:
    #     axs[0, 0].plot(x, e)
    #     axs[0, 1].plot(x, e)
    # axs[0, 0].set_title('Expertise')
    # axs[0, 0].set_ylim([0, 1])
    # axs[0, 0].set_yticks(np.arange(0, 1, step=0.1))
    # axs[0, 1].set_title('Expertise (zoom)')
    # for g in sub_g:
    #     axs[1, 0].plot(x, g)
    #     axs[1, 1].plot(x, g)
    # axs[1, 0].set_title('Goodwill')
    # axs[1, 0].set_ylim([0, 1])
    # axs[1, 0].set_yticks(np.arange(0, 1, step=0.1))
    # axs[1, 1].set_title('Goodwill (zoom)')
    #
    # for c in sub_c:
    #     axs[2, 0].plot(x, c)
    #     axs[2, 1].plot(x, c)
    # axs[2, 0].set_title('Coherence behaviour')
    # axs[2, 0].set_ylim([0, 1])
    # axs[2, 0].set_yticks(np.arange(0, 1, step=0.1))
    # axs[2, 1].set_title('Coherence behaviour (zoom)')
    #
    # for t in sub_t:
    #     axs[3, 0].plot(x, t)
    #     axs[3, 1].plot(x, t)
    # axs[3, 0].set_title('Trust')
    # axs[3, 0].set_ylim([0, 1])
    # axs[3, 0].set_yticks(np.arange(0, 1, step=0.1))
    # axs[3, 1].set_title('Trust (zoom)')
    #
    # fig.tight_layout(pad=3)
    # fig.suptitle(s, size=14)
    # plt.figure(figsize=(18, 14), dpi=80)
    # for ax in axs.flat:
    #     ax.set(xlabel='News requests during the time', ylabel='Metric value')
    #     ax.legend(df['Topic'].unique(), loc='lower right')
    fig, axs = plt.subplots(4, 1, figsize=(12, 12))
    x = np.arange(0, len(sub_et))
    for e in sub_e:
        axs[0].plot(x, e)
    axs[0].set_title('Expertise')
    axs[0].set_ylim([0, 1])
    axs[0].set_yticks(np.arange(0, 1, step=0.1))
    for g in sub_g:
        axs[1].plot(x, g)
    axs[1].set_title('Goodwill')
    axs[1].set_ylim([0, 1])
    axs[1].set_yticks(np.arange(0, 1, step=0.1))


    for c in sub_c:
        axs[2].plot(x, c)
    axs[2].set_title('Coherence behaviour')
    axs[2].set_ylim([0, 1])
    axs[2].set_yticks(np.arange(0, 1, step=0.1))

    for t in sub_t:
        axs[3].plot(x, t)
    axs[3].set_title('Trust')
    axs[3].set_ylim([0, 1])
    axs[3].set_yticks(np.arange(0, 1, step=0.1))

    fig.tight_layout(pad=3, rect=[0, 0, 0.75, 1])
    fig.suptitle(s, size=14, y=0.99)
    plt.figure(figsize=(20, 20), dpi=80)
    for ax in axs.flat:
        ax.set(xlabel='News requests during the time', ylabel='Metric value')
        ax.legend(df['Topic'].unique(), loc='center left', bbox_to_anchor=(1, 0.5))
        # ax.legend(df['Topic'].unique(), loc='lower right')
    plt.show()
