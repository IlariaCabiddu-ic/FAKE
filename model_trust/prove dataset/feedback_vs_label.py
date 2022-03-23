"""this file show the difference between trust model with feedback and label """
import pandas as pd
import numpy as np
import warnings
from pandas.core.common import SettingWithCopyWarning
import matplotlib.pyplot as plt
from itertools import chain
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
df = df.rename(columns={"source": "Source", "label": "label", "message-based": "Message_based", "mainTopic": "Topic"})
df["Feedback"] = df["Message_based"]  # copy of message based value
df['ID'] = pd.Series(range(0, df.shape[0]-1))
datetimes = pd.Series(random_dates(start, end, df.shape[0]))  # sorted datetimes insertion
datetimes = datetimes.sort_values(ascending=True)
datetimes = datetimes.reset_index(drop=True)
df['Datetime'] = datetimes

"-----------------analysis of each present source--------------"
df = df[df['Source'] == 'nytimes']  # selection of one source and one topic
sources = list(df['Source'].unique())
topics = list(df['Topic'].unique())
for s in sources:
    ordered_trust = {}
    ordered_trust_label = {}
    print("source ", s)
    sub_et = []
    sub_gt = []
    sub_ct = []
    sub_tt = []
    sub_e = []
    sub_g = []
    sub_c = []
    sub_t = []
    sub_g_lab = []
    sub_c_lab = []
    sub_t_lab = []
    sube = []
    subg = []
    subc = []
    subt = []
    E = []
    G = []
    C = []
    T = []
    Glab = []
    Clab = []
    Tlab = []
    sub_df = df[df['Source'] == s]  # iteration for the each source
    err =[]
    for j in range(sub_df.shape[0]):
        sub_d = sub_df[0:j+1]  # application of trust model for each news entry
        sub_topics = list(sub_d['Topic'].unique())
        expertise = model_trust.compute_expertise(df=sub_d, topics=sub_topics)
        exp = list(expertise.values())

        relevance = model_trust.compute_relevance(df=sub_d, topics=sub_topics)

        goodwill = model_trust.compute_goodwill(df=sub_d, topics=sub_topics, relevance=relevance)
        good = list(goodwill.values())
        goodwill_label = model_trust.compute_goodwill_with_label(df=sub_d, topics=sub_topics, relevance=relevance)
        good_label = list(goodwill_label.values())

        coherence = model_trust.compute_coherence(df=sub_d, topics=sub_topics)
        coh = list(coherence.values())
        coherence_label = model_trust.compute_coherence_with_label(df=sub_d, topics=sub_topics)
        coh_label = list(coherence_label.values())

        trust = model_trust.compute_trust(expertise=expertise, goodwill=goodwill,
                                          coherence=coherence, topics=sub_topics)

        trust_label = model_trust.compute_trust(expertise=expertise, goodwill=goodwill_label,
                                                coherence=coherence_label, topics=sub_topics)

        if len(sub_topics) < len(topics):
            for t_idx in topics:
                # sub_df_topic = sub_df[sub_df['Topic'] == t_idx]
                # sub_df_topic['Feedback'] = (1+sub_df_topic ['Feedback'])/2
                # print(s,t_idx, sub_df_topic['Feedback'].mean())  # feedback's mean
                if t_idx not in trust.keys():  # if the trust doens't exist, it is equal to 0.2
                    trust[t_idx] = 0.2
                if t_idx not in trust_label.keys():  # same for trust calculated with labels
                    trust_label[t_idx] = 0.2
                else:
                    pass

        for item in topics:  # sorting of value based on the topic
            ordered_trust[item] = trust[item]
            ordered_trust_label[item] = trust_label[item]
        tru = list(ordered_trust.values())
        tru_label = list(ordered_trust_label.values())
        E.append(exp)
        G.append(good)
        C.append(coh)
        T.append(tru)
        Glab.append(good_label)
        Clab.append(coh_label)
        Tlab.append(tru_label)
    sub_et = utils.reorder_metrics(df=df, metric=E)
    sub_gt = utils.reorder_metrics(df=df, metric=G)
    sub_ct = utils.reorder_metrics(df=df, metric=C)
    sub_tt = utils.reorder_metrics(df=df, metric=T)

    sub_gt_lab = utils.reorder_metrics(df=df, metric=Glab)
    sub_ct_lab = utils.reorder_metrics(df=df, metric=Clab)
    sub_tt_lab = utils.reorder_metrics(df=df, metric=Tlab)

    for k in range(len(df['Topic'].unique())):
        sub_e.append(list(chain(sub_exp[k] for sub_exp in sub_et)))
        sub_g.append(list(chain(sub_good[k] for sub_good in G)))
        sub_c.append(list(chain(sub_coh[k] for sub_coh in C)))
        sub_t.append(list(chain(sub_tru[k] for sub_tru in T)))

        sub_g_lab.append(list(chain(sub_good[k] for sub_good in Glab)))
        sub_c_lab.append(list(chain(sub_coh[k] for sub_coh in Clab)))
        sub_t_lab.append(list(chain(sub_tru[k] for sub_tru in Tlab)))

    for k in range(len(sub_t)):  # compute error mean (between feedback and label)
        err.append(sum(abs(np.array(sub_t[k]) - np.array(sub_t_lab[k])))/len(sub_t[k]))

    fig, axs = plt.subplots(3, 3, figsize=(12, 12))  # plot trust with feedback and with labels

    for t in sub_t:
        xt = np.arange(0, len(t))
        axs[int(sub_t.index(t)/3), sub_t.index(t) % 3].plot(xt, t)
    for t_lab in sub_t_lab:
        xt = np.arange(0, len(t_lab))
        axs[int(sub_t_lab.index(t_lab)/3), sub_t_lab.index(t_lab) % 3].plot(xt, t_lab)

    fig.tight_layout(pad=3)
    fig.suptitle(s, size=14, y=1)
    plt.figure(figsize=(20, 20), dpi=80)
    topic = df['Topic'].unique()
    for ax, index in zip(axs.flat, topics):

        ax.set(xlabel='News requests during the time', ylabel='Trust value')
        ax.legend(['feedback', 'label'], loc='best')
        ax.set_title(index)
        ax.set_ylim([0, 1])
        ax.set_yticks(np.arange(0, 1, step=0.1))
    plt.show()

