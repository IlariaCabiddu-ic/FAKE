import pandas as pd
import numpy as np
import warnings
from pandas.core.common import SettingWithCopyWarning
import matplotlib.pyplot as plt
from itertools import chain,groupby
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
# topics = pd.Series(['Economics','Politics', 'Tech', 'Actuality'])
# topics= topics.repeat((df.shape[0]/topics.shape[0])+1)
# topics = topics.reset_index(drop=True)
# topics = topics[0:df.shape[0]]
# df['Topic'] = topics[0:df.shape[0]]
# df = df.sample(frac=1)
df['ID'] = pd.Series(range(0,df.shape[0]-1))
datetimes = pd.Series(random_dates(start, end, df.shape[0]))
datetimes = datetimes.sort_values(ascending=True)
datetimes = datetimes.reset_index(drop=True)
df['Datetime'] = datetimes
dfupdate = df[df['Source'] == 'nytimes']
dfupdate= dfupdate.reset_index(drop=True)

df1 = dfupdate[dfupdate['Topic'] == 'Politics']  # 170 news
# df1 = dfupdate[dfupdate['Topic'] == 'Crime & security']  # 104 news
# df1 = dfupdate[dfupdate['Topic'] == 'Business & Economy']  # 80 news

df1 = df1.reset_index(drop=True)

perc = [0.1,0.2,0.3]
n_iteration = 10
T = []
Tlab = []
"""---computation of trust with label"""
for j in range(df1.shape[0]):
    sub_d = df1[0:j + 1]
    sub_topics = list(sub_d['Topic'].unique())
    expertise1 = model_trust.compute_expertise(df=sub_d, topics=sub_topics)
    exp1 = list(expertise1.values())

    relevance1 = model_trust.compute_relevance(df=sub_d, topics=sub_topics)
    goodwill1 = model_trust.compute_goodwill(df=sub_d, topics=sub_topics, relevance=relevance1)
    good1 = list(goodwill1.values())
    goodwill_label = model_trust.compute_goodwill_with_label(df=sub_d, topics=sub_topics, relevance=relevance1)
    good_label = list(goodwill_label.values())
    coherence1 = model_trust.compute_coherence(df=sub_d, topics=sub_topics)
    coh1 = list(coherence1.values())
    coherence_label = model_trust.compute_coherence_with_label(df=sub_d, topics=sub_topics)
    coh_label = list(coherence_label.values())
    trust1 = model_trust.compute_trust(expertise=expertise1, goodwill=goodwill1,
                                      coherence=coherence1, topics=sub_topics)
    # tru1 = list(trust1.values())
    trust_label = model_trust.compute_trust(expertise=expertise1, goodwill=goodwill_label,
                                            coherence=coherence_label, topics=sub_topics)
    # tru_label = list(trust_label.values())
    T.append(*trust1.values())
    Tlab.append(*trust_label.values())

x = np.arange(0, df1.shape[0])
plt.plot(x, Tlab, 'r--')
plt.plot(x,T)

"""----modification of dataset------"""
for k in perc:
    overall_df = pd.DataFrame()
    overall_exp = pd.Series()
    overall_rel = pd.Series()
    overall_good = pd.Series()
    overall_coh = pd.Series()
    overall_tru = pd.Series()
    for i in range(n_iteration):
        df1new = df1.copy()
        rand = np.random.choice(df1.shape[0],size=int(k * len(df1)), replace=False)
        rand = list(rand)
        df1update = df1.iloc[rand]  # take percentage of df in order to change values
        # df1update['Message_based'] = -df1update['Message_based']
        # df1update['Feedback'] = -df1update['Feedback']
        df1update['label'] = -df1update['label']
        df1new.loc[df1update.index, :] = df1update[:]
        overall_df = pd.concat([overall_df,df1new])
        overall_df = overall_df.reset_index(drop=True)
        for j in range (df1.shape[0]):
            sub_df1 = overall_df[0:j%df1.shape[0] + 1]
            expertise = model_trust.compute_expertise(sub_df1,sub_df1['Topic'].unique())
            overall_exp = pd.concat([overall_exp,pd.Series(expertise.values())])
            overall_exp = overall_exp.reset_index(drop=True)

            relevance = model_trust.compute_relevance(sub_df1, list(sub_df1['Topic'].unique()))
            overall_rel = pd.concat([overall_rel, pd.Series(relevance.values())])
            overall_rel = overall_rel.reset_index(drop=True)

            # goodwill = model_trust.compute_goodwill(sub_df1, list(sub_df1['Topic'].unique()),relevance)
            goodwill = model_trust.compute_goodwill_with_label(sub_df1,list(sub_df1['Topic'].unique()),relevance)
            overall_good = pd.concat([overall_good, pd.Series(goodwill.values())])
            overall_good = overall_good.reset_index(drop=True)

            # coherence = model_trust.compute_coherence(sub_df1, list(sub_df1['Topic'].unique()))
            coherence = model_trust.compute_coherence_with_label(sub_df1, list(sub_df1['Topic'].unique()))
            overall_coh = pd.concat([overall_coh, pd.Series(coherence.values())])
            overall_coh = overall_coh.reset_index(drop=True)

            trust = model_trust.compute_trust(expertise, goodwill, coherence, list(sub_df1['Topic'].unique()))
            overall_tru = pd.concat([overall_tru, pd.Series(trust.values())])
            overall_tru = overall_tru.reset_index(drop=True)
    final_trust = np.zeros(df1.shape[0])
    overall_tru = overall_tru.to_numpy()

    for h in range(len(overall_tru)):
        final_trust[h%df1.shape[0]] = final_trust[h%df1.shape[0]] + overall_tru[h]  # circular indexing
    final_trust = final_trust/n_iteration
    plt.plot(x,final_trust,'--')
plt.xlabel('Number Politic news')
plt.ylabel('Trust value')
plt.axis([0, 170, 0, 1])
# plt.legend(['Trust with label','trust with feedback',
#             'trust with 10% of feedback error','trust with 20% of feedback error',
#             'trust with 30% of feedback error'], loc='lower right')
plt.legend(['Trust with label','trust with feedback',
            'trust with 10% of label error','trust with 20% of label error',
            'trust with 30% of label error'], loc='lower right')
plt.show()
