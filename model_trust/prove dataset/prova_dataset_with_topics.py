import pandas as pd
import numpy as np
import warnings
from pandas.core.common import SettingWithCopyWarning
import matplotlib.pyplot as plt
from itertools import chain,groupby
import model_trust
import utils

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def random_dates(start, end, n):  # generate random date for the dataset's datetimes
    start_u = start.value // 10 ** 9
    end_u = end.value // 10 ** 9
    return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s')


start = pd.to_datetime('2018-01-01')  # start date
end = pd.to_datetime('2022-02-01')  # end date

dataframe = pd.read_csv('../Dataset/TEST_01_with_topics.csv')
df = dataframe[["source", "label", "message-based","mainTopic"]]
df = df.rename(columns={"source":"Source", "label":"label", "message-based":"Message_based", "mainTopic":"Topic"})
df["Feedback"] = df["Message_based"]  # copy of message based value
df['ID'] = pd.Series(range(0,df.shape[0]-1))
datetimes = pd.Series(random_dates(start, end, df.shape[0]))
datetimes = datetimes.sort_values(ascending=True)   # sorted datetimes insertion
datetimes = datetimes.reset_index(drop=True)
df['Datetime'] = datetimes
# df = df.sort_values(by="Datetime")
# df = df[df['Source'] == 'nytimes']


"-----------------analysis of each present source--------------"

sources = list(df['Source'].unique())
topics = list(df['Topic'].unique())

for s in sources:
    ordered_trust = {}
    print("source ",s)
    sub_et = []
    sub_gt = []
    sub_ct = []
    sub_tt = []
    sub_e = []
    sub_g = []
    sub_c = []
    sub_t = []
    sube = []
    subg = []
    subc = []
    subt = []
    E = []
    G = []
    C = []
    T = []
    sub_df = df[df['Source'] == s]  # iteration for the each source
    for j in range(sub_df.shape[0]):
        sub_d = sub_df[0:j+1]  # application of trust model for each news entry
        # print(j)
        sub_topics = list(sub_d['Topic'].unique())
        expertise = model_trust.compute_expertise(df=sub_d, topics=sub_topics)
        exp = list(expertise.values())

        relevance = model_trust.compute_relevance(df=sub_d, topics=sub_topics)
        goodwill = model_trust.compute_goodwill(df=sub_d, topics=sub_topics, relevance=relevance)
        good = list(goodwill.values())
        coherence = model_trust.compute_coherence(df=sub_d, topics=sub_topics)
        coh = list(coherence.values())
        trust = model_trust.compute_trust(expertise=expertise, goodwill=goodwill,
                                          coherence=coherence, topics=sub_topics)
        if len(sub_topics) < len(topics):
            for t_idx in topics:
                # sub_df_topic = sub_df[sub_df['Topic'] == t_idx]
                # sub_df_topic['Feedback'] = (1+sub_df_topic ['Feedback'])/2
                # print(s,t_idx, sub_df_topic['Feedback'].mean())  # take a feedback's mean
                if t_idx not in trust.keys():  # if the trust doens't exist, it is equal to 0.2
                    trust[t_idx] = 0.2
                else:
                    pass
        for item in topics:
            ordered_trust[item] = trust[item]
        tru = list(ordered_trust.values())
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

    for h in range(len(sub_e)):
        sube.append(list(key for key, _ in groupby(sub_e[h])))
        subg.append(list(key for key, _ in groupby(sub_g[h])))
        subc.append(list(key for key, _ in groupby(sub_c[h])))
        # subt.append(list(key for key, _ in groupby(sub_t[h])))
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

    for e in sub_e:
        xe = np.arange(0, len(e))
        axs[0].plot(xe, e)
    axs[0].set_title('Expertise')
    axs[0].set_ylim([0, 1])
    axs[0].set_yticks(np.arange(0, 1, step=0.1))
    for g in subg:
        xg = np.arange(0, len(g))
        axs[1].plot(xg, g)
    axs[1].set_title('Goodwill')
    axs[1].set_ylim([0, 1])
    axs[1].set_yticks(np.arange(0, 1, step=0.1))


    for c in subc:
        xc = np.arange(0, len(c))
        axs[2].plot(xc, c)
    axs[2].set_title('Coherence behaviour')
    axs[2].set_ylim([0, 1])
    axs[2].set_yticks(np.arange(0, 1, step=0.1))

    for t in sub_t:
        xt = np.arange(0, len(t))
        axs[3].plot(xt, t)
    axs[3].set_title('Trust')
    axs[3].set_ylim([0, 1])
    axs[3].set_yticks(np.arange(0, 1, step=0.1))

    fig.tight_layout(pad=3, rect=[0, 0, 0.75, 1])
    fig.suptitle(s, size=14, y=0.99)
    plt.figure(figsize=(20, 20), dpi=80)
    for ax in axs.flat:
        ax.set(xlabel='News requests during the time', ylabel='Metric value')
        ax.legend(sub_topics, loc='center left', bbox_to_anchor=(1, 0.5))
        # ax.legend(df['Topic'].unique(), loc='lower right')
    plt.show()
