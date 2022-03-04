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
df["Feedback"] = df["Message_based"]
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
        sub_d = sub_df[0:j+1]
        sub_topics = list(sub_d['Topic'].unique())
        expertise = model_trust.compute_expertise(df=sub_d, topics=sub_topics)
        exp = list(expertise.values())
        # print("The expertise of analysed source '", s, "' is: ", expertise)

        relevance = model_trust.compute_relevance(df=sub_d, topics=sub_topics)
        # print("The relevance of analysed source '", s, "' is: ", relevance)
        goodwill = model_trust.compute_goodwill(df=sub_d, topics=sub_topics, relevance=relevance)
        good = list(goodwill.values())
        goodwill_label = model_trust.compute_goodwill_with_label(df=sub_d, topics=sub_topics, relevance=relevance)
        good_label = list(goodwill_label.values())
        # print("The goodwill of analysed source '", s, "' is: ", goodwill)
        coherence = model_trust.compute_coherence(df=sub_d, topics=sub_topics)
        coh = list(coherence.values())
        coherence_label = model_trust.compute_coherence_with_label(df=sub_d, topics=sub_topics)
        coh_label = list(coherence_label.values())
        # print("The coherence of analysed source '", s, "' is: ", coherence)
        trust = model_trust.compute_trust(expertise=expertise, goodwill=goodwill,
                                          coherence=coherence, topics=sub_topics)
        tru = list(trust.values())
        trust_label = model_trust.compute_trust(expertise=expertise, goodwill=goodwill_label,
                                          coherence=coherence_label, topics=sub_topics)
        tru_label = list(trust_label.values())
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

    for k in range(len(sub_t)):
        err.append(sum(abs(np.array(sub_t[k]) - np.array(sub_t_lab[k])))/len(sub_t[k]))
    # for h in range(len(sub_e)):
    #     sube.append(list(key for key, _ in groupby(sub_e[h])))
    #     subg.append(list(key for key, _ in groupby(sub_g[h])))
    #     subc.append(list(key for key, _ in groupby(sub_c[h])))
    #     subt.append(list(key for key, _ in groupby(sub_t[h])))
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
    # fig, axs = plt.subplots(4, 1, figsize=(12, 12))
    #
    # for e in sub_e:
    #     xe = np.arange(0, len(e))
    #     axs[0].plot(xe, e)
    # axs[0].set_title('Expertise')
    # axs[0].set_ylim([0, 1])
    # axs[0].set_yticks(np.arange(0, 1, step=0.1))
    # for g in sub_g:
    #     xg = np.arange(0, len(g))
    #     axs[1].plot(xg, g)
    # axs[1].set_title('Goodwill')
    # axs[1].set_ylim([0, 1])
    # axs[1].set_yticks(np.arange(0, 1, step=0.1))
    #
    #
    # for c in sub_c:
    #     xc = np.arange(0, len(c))
    #     axs[2].plot(xc, c)
    # axs[2].set_title('Coherence behaviour')
    # axs[2].set_ylim([0, 1])
    # axs[2].set_yticks(np.arange(0, 1, step=0.1))
    #
    # for t in sub_t:
    #     xt = np.arange(0, len(t))
    #     axs[3].plot(xt, t)
    # axs[3].set_title('Trust')
    # axs[3].set_ylim([0, 1])
    # axs[3].set_yticks(np.arange(0, 1, step=0.1))
    #
    # fig.tight_layout(pad=3, rect=[0, 0, 0.75, 1])
    # fig.suptitle(s, size=14, y=0.99)
    # plt.figure(figsize=(20, 20), dpi=80)
    # for ax in axs.flat:
    #     ax.set(xlabel='News requests during the time', ylabel='Metric value')
    #     ax.legend(df['Topic'].unique(), loc='center left', bbox_to_anchor=(1, 0.5))
    #     # ax.legend(df['Topic'].unique(), loc='lower right')
    # plt.show()
    fig, axs = plt.subplots(3,3, figsize=(12, 12))

    for t in sub_t:
        xt = np.arange(0, len(t))
        axs[sub_t.index(t)%3,int(sub_t.index(t)/3)].plot(xt, t)
    for t_lab in sub_t_lab:
        xt = np.arange(0, len(t_lab))
        axs[sub_t_lab.index(t_lab)%3,int(sub_t_lab.index(t_lab)/3)].plot(xt, t_lab)


    fig.tight_layout(pad=3)
    fig.suptitle(s, size=14, y=1)
    plt.figure(figsize=(20, 20), dpi=80)
    topic=df['Topic'].unique()
    for ax,index in zip(axs.flat,topic):

        ax.set(xlabel='News requests during the time', ylabel='Trust value')
        ax.legend(['feedback','label'], loc='best')
        ax.set_title(index)
        ax.set_ylim([0, 1])
        ax.set_yticks(np.arange(0, 1, step=0.1))
        # ax.legend(df['Topic'].unique(), loc='lower right')
    plt.show()

