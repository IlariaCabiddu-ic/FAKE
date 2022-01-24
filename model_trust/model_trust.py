'''-----------Model trust------------'''
import collections
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import geom
import numpy as np
import seaborn as sns


def compute_expertise(df, topics):
    alfa = 0.5
    beta = 0.5
    Q_st = []
    dictionary_Q = {}
    dictionary_E = {}

    '''computation of M_st that is a percentage number of published news items 
        of a specific topic compared to the total number of news items of a source'''
    df = df.drop_duplicates()
    "Compute focus theme M_st"
    M_st = df['Topic'].to_list()
    M_st = dict(collections.Counter(M_st))
    "Number of total news"
    N_s = sum(M_st.values())
    for i in M_st:
        M_st[i] = float(round(M_st[i]/N_s, 4))

    "Compute technicality Q_st"
    for t in topics:
        df_sub = df[df['Topic'] == t]  # iteration for the each topic
        q = list(df_sub['Message_based'])
        # sns.distplot(q, hist=False)
        # plt.ylabel("Pdf", fontsize="18")
        # plt.xlabel("Message based value", fontsize="18")
        # plt.title("Technicality distribution for politics", fontsize="18")
        # plt.show()
        Q_st.append(sum(df_sub['Message_based']) / (df_sub.shape[0]))  # divide by number of topic news P_st
        dictionary_Q[t] = sum(df_sub['Message_based']) / (df_sub.shape[0])
    zipped_lists = zip(list(M_st.values()), Q_st)
    expertise = [alfa * x + beta*y for (x, y) in zipped_lists]
    expertise = [round(x, 2) for x in expertise]
    for t in range(len(topics)):
        dictionary_E[topics[t]] = expertise[t]
    # keys = dictionary_E.keys()
    # values = dictionary_E.values()

    # plt.bar(keys, values)
    # plt.ylabel("Expertise value", fontsize="18")
    # plt.xlabel("Topics", fontsize="18")
    # plt.show()
    return dictionary_E


def compute_relevance(df):
    dictionary_r = dict(zip(df['ID'].unique(), df.value_counts(['ID'])))
    relevance = {k: v / total for total in (sum(dictionary_r.values()),)
                 for k, v in dictionary_r.items()}  # normalization values
    return relevance


def compute_goodwill(df, topics, relevance):
    G_st = []
    dictionary_G = {}
    df_unique = df.drop_duplicates()
    df_unique['Relevance'] = df_unique['ID'].map(relevance)
    for t in topics:
        df_sub = df_unique[df_unique['Topic'] == t]  # iteration for the each topic
        g = round(sum(df_sub['Relevance'] * df_sub['Feedback']), 4)
        G_st.append(0.5*(1+(g / (df_sub.shape[0]))))  # divide by number of topic news P_st
        dictionary_G[t] = 0.5*(1+(g / (df_sub.shape[0])))
    return dictionary_G


def compute_coherence(df, topics):
    C_st = []
    dictionary_C = {}
    df_unique = df.drop_duplicates()
    datetimes = pd.to_datetime(df_unique['Datetime'])
    df_unique['Datetime'] = datetimes
    for t in topics:
        df_sub = df_unique[df_unique['Topic'] == t]  # iteration for the each topic
        df_sub = df_sub.set_index('Datetime')
        df_sub = df_sub.sort_index(ascending=False)
        # print(df_sub)
        samples = np.arange(1, df_sub.shape[0]+1)
        p = 0.8
        # Calculate geometric probability distribution (WITH THE FINITE UPPER BOUND)
        weight_l = geom.pmf(samples, p)
        res = (1-sum(weight_l)) / len(weight_l)  # in order to have a unitary area
        weight_l = weight_l + res
        df_sub['Weight_l'] = weight_l
        # print(df_sub)
        # Plot the probability distribution
        # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # ax.plot(samples, weight_l, 'bo', ms=8, label='geom pmf')
        # plt.ylabel("Probability", fontsize="18")
        # plt.xlabel("Samples", fontsize="18")
        # plt.title("Geometric Distribution", fontsize="18")
        # plt.stem(samples,weight_l)
        # plt.show()
        # print(weight_l)
        c = list(df_sub['Weight_l'] * df_sub['Feedback'])
        # print(t, h)
        # plt.plot(c, linestyle='dotted')
        # plt.ylabel("Coherence values", fontsize="18")
        # plt.xlabel("Coherence news samples", fontsize="18")
        # plt.title(("Coherence plot", t), fontsize="18")
        # plt.show()
        C_st.append(0.5*(1+(round(sum(df_sub['Weight_l'] * df_sub['Feedback']), 4))))  # divide by number of topic news P_st
        dictionary_C[t] = 0.5*(1+(round(sum(df_sub['Weight_l'] * df_sub['Feedback']), 4)))

    return dictionary_C


def compute_trust(expertise, goodwill, coherence, topics):
    trust = []
    coherence_new = []
    dictionary_T = {}
    # for t in topics:
    #     l = list(coherence[t])
    #     hist = sum(l)/len(l)
    #     coherence_new.append(hist)
    # print(coherence_new)
    sigma = 0.05  # weight for expertise metric
    omega = 0.9  # weight for coherence metric
    gamma = 0.05 # weight for goodwill metric
    e = [x * sigma for x in list(expertise.values())]
    g = [x * gamma for x in list(goodwill.values())]
    h = [x * omega for x in list(coherence.values())]
    for i in range(len(topics)):
        trust.append(e[i] + g[i] + h[i])

    for t in range(len(topics)):
        dictionary_T[topics[t]] = trust[t]
    return dictionary_T