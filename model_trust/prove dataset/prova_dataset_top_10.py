"""This file is used to create a ranking of best sources for each topic.
Moreover, the trust value in relation to the number of fake news stories
and the average feedback is taken into account """
import pandas as pd
import numpy as np
import warnings
from pandas.core.common import SettingWithCopyWarning
import model_trust
import utils

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def random_dates(start, end, n):  # generate random date for the dataset's datetimes
    start_u = start.value // 10 ** 9
    end_u = end.value // 10 ** 9
    return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s')


start = pd.to_datetime('2018-01-01')  # start date
end = pd.to_datetime('2022-02-01')  # end date
flag = 1
dataframe = pd.read_csv('../Dataset/TEST_01_with_topics.csv')
df = dataframe[["source", "label", "message-based","mainTopic"]]
df = df.rename(columns={"source":"Source", "label":"label", "message-based":"Message_based", "mainTopic":"Topic"})
df["Feedback"] = df["Message_based"]  # copy of message based value
df['ID'] = pd.Series(range(0,df.shape[0]-1))
datetimes = pd.Series(random_dates(start, end, df.shape[0]))
datetimes = datetimes.sort_values(ascending=True)   # sorted datetimes insertion
datetimes = datetimes.reset_index(drop=True)
df['Datetime'] = datetimes



"-----------------analysis of each present source--------------"

sources = list(df['Source'].unique())
topics = list(df['Topic'].unique())
cols = pd.MultiIndex.from_product([topics, ['Trust', '% Fake news']])
df_trust_fake = pd.DataFrame(index=sources,columns=cols)
df_trust = pd.DataFrame(index=sources, columns=topics)
df_top = pd.DataFrame()
df_top10 = pd.DataFrame()
ordered_trust = {}
for s in sources:  # selection of source
    count_fake = []
    print("source ", s)
    sub_et = []
    sub_gt = []
    sub_ct = []
    sub_tt = []
    # sub_e = []
    # sub_g = []
    # sub_c = []
    # sub_t = []
    # sube = []
    # subg = []
    # subc = []
    # subt = []
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

        tru = list(trust.values())
        E.append(exp)
        G.append(good)
        C.append(coh)
        T.append(tru)
    sub_et = utils.reorder_metrics(df=df, metric=E)
    sub_gt = utils.reorder_metrics(df=df, metric=G)
    sub_ct = utils.reorder_metrics(df=df, metric=C)
    sub_tt = utils.reorder_metrics(df=df, metric=T)
    if len(sub_topics)<len(topics):
        for t_idx in topics:
            # sub_df_topic = sub_df[sub_df['Topic'] == t_idx]
            # sub_df_topic['Feedback'] = (1+sub_df_topic ['Feedback'])/2
            # print(s,t_idx, sub_df_topic['Feedback'].mean())
            if t_idx not in trust.keys():  # if the trust doens't exist, it is equal to 0.2
                trust[t_idx]=0.2
            else:
                pass
    for item in topics:
        ordered_trust[item] = trust[item]
    for t in topics:  # take the percentage of fake news of the topic t (or number fake news/number news )
        sub_dfake =np.where((sub_df['Topic'] == t) & (sub_df['label'] == -1))[0]

        if len(sub_df[sub_df['Topic'] == t]) != 0:
            count_fake.append((len(sub_dfake))/len(sub_df[sub_df['Topic'] == t]))  # tuple([len(sub_dfake),len(sub_df[sub_df['Topic'] == t])])) per prendere la frazione
        else:
            count_fake.append('nan')
        # print(len(sub_dfake), len(sub_df[sub_df['Topic'] == t]), t,((len(sub_dfake))/len(sub_df[sub_df['Topic'] == t]) ))
        # print(count_fake)
    trust_list = [*ordered_trust.values()]
    result = [None]*(len(trust_list)+len(count_fake))
    result[::2] = trust_list
    result[1::2] = count_fake

    df_trust_fake.loc[s] = result
    df_trust.loc[s] = trust_list
# print(df_trust.head(100).to_string())

for col in df_trust:
    df_top = df_trust[col].copy()
    df_top10[col] = df_top.sort_values(ascending=False).index
df_top10 = df_top10.iloc[0:9,:]
df_top10.index += 1
print(df_top10.head(30).to_string())  # top 10 of best sources for each topic

