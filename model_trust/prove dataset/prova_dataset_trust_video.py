"""In this file the trust model is tested by the using a part of a kaggle dataset"""
import pandas as pd
import numpy as np
import warnings
from pandas.core.common import SettingWithCopyWarning
import matplotlib.pyplot as plt
from itertools import chain,groupby
import model_trust
import utils

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    # for rect in rects:
    #     height = rect.get_height()
    #     plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,
    #             '%d' % int(height),
    #             ha='center', va='bottom')
    # For each bar: Place a label
    for rect in rects:
        # Get X and Y placement of label from rect.
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2

        # Number of points between bar and label. Change to your liking.
        space = 5
        # Vertical alignment for positive values
        ha = 'left'

        # If value of bar is negative: Place label left of bar
        if x_value < 0:
            # Invert space to place label to the left
            space *= -1
            # Horizontally align label at right
            ha = 'right'

        # Use X value as label and format number with one decimal place
        label = "{:.1f}".format(x_value)

        # Create annotation
        plt.annotate(
            label,  # Use `label` as label
            (x_value, y_value),  # Place label at end of the bar
            xytext=(space, 0),  # Horizontally shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            va='center',  # Vertically center label
            ha=ha)  # Horizontally align label differently for
        # positive and negative values.


warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

font = {'family': 'DejaVu Sans',
            'weight': 'bold',
            'size': 22}


def random_dates(start, end, n):  # generate random date for the dataset's datetimes
    start_u = start.value // 10 ** 9
    end_u = end.value // 10 ** 9
    return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s')


start = pd.to_datetime('2018-01-01')  # start date
end = pd.to_datetime('2022-02-01')  # end date
rand_list = []
dataframe = pd.read_csv('../Dataset/df_complete_new_sent.csv')
data1 = pd.read_csv('../Dataset/TEST_01_with_topics.csv')
df0 = pd.concat([data1[["source", "link"]], dataframe], axis=1)
df1 = df0.loc[ (df0.source == 'breitbart') |(df0.source == 'nytimes') |(df0.source == 'newsgloble24')]
df1 = df1.loc[ (df0.mainTopic == 'Politics') |(df0.mainTopic == 'Business & Economy') |(df0.mainTopic == 'Crime & security')]
df2 = df1.loc[(df0.source == 'newsgloble24')]

# df2 = df1.loc[1440:1528,:].copy()
# df2.at[1460, 'message-based'] = -df2.at[1460, 'message-based']
# df2.at[1468, 'message-based'] = -df2.at[1468, 'message-based']
# df2.at[1474, 'message-based'] = -df2.at[1474, 'message-based']
# df2.at[1478, 'message-based'] = -df2.at[1478, 'message-based']
# df2.at[1482, 'message-based'] = -df2.at[1482, 'message-based']
# df2.at[1489, 'message-based'] = -df2.at[1489, 'message-based']
# df2.at[1521, 'message-based'] = -df2.at[1521, 'message-based']
df3 = df1.loc[(df0.source == 'breitbart')]
df3 = df3.loc[1440:1564,:]
df4 = df1.loc[(df0.source == 'nytimes')]
df4 = df4.loc[1440:1550,:]
dfupload = pd.concat([df1, df2,df3,df4], axis=0).drop_duplicates(keep='last')
df2.at[273, 'message-based'] = - df2.at[273, 'message-based']
df2.at[322, 'message-based'] = - df2.at[322, 'message-based']
# dfupload.to_csv("slice_csv.csv")

df = dfupload[["source", "label", "message-based","mainTopic","index"]]
df = df.rename(columns={"source":"Source", "label":"label", "message-based":"Message_based",
                              "mainTopic":"Topic", "index":"ID"})
df["Feedback"] = df["Message_based"]  # copy of message based value
# df['ID'] = pd.Series(range(0,df.shape[0]-1))
datetimes = pd.Series(random_dates(start, end, df.shape[0]))
datetimes = datetimes.sort_values(ascending=True)   # sorted datetimes insertion
datetimes = datetimes.reset_index(drop=True)
df['Datetime'] = datetimes
sources = list(df['Source'].unique())
topics = list(df['Topic'].unique())
sub_e = []
sub_g = []
sub_c = []
sub_t = []
for s in sources:
    ordered_trust = {}
    print("source ",s)
    sub_et = []
    sub_gt = []
    sub_ct = []
    sub_tt = []
    # sub_e = []
    # sub_g = []
    # sub_c = []
    # sub_t = []
    sube = []
    subg = []
    subc = []
    subt = []
    E = []
    G = []
    C = []
    T = []
    sub_df = df[df['Source'] == s]  # iteration for the each source
    if len(sub_df)>10:
        print(len(sub_df))
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

    for h in range(len(sub_e)):  # in order to have unique values
        # sube.append(list(key for key, _ in groupby(sub_e[h])))
        subg.append(list(key for key, _ in groupby(sub_g[h])))
        subc.append(list(key for key, _ in groupby(sub_c[h])))
        # subt.append(list(key for key, _ in groupby(sub_t[h])))
slice = [20,2,28]
jump = [9,18,27]
t_breitbart = np.array(sub_t[0:3])
t_breitbart = (t_breitbart[:,-20:]).T
t_NYT = np.array(sub_t[3:6])
t_NYT = t_NYT[:,-20:].T
t_newglobe = np.array(sub_t[6:9])
t_newglobe = (t_newglobe[:,-20:]).T
# df_breit = pd.DataFrame(t_breitbart, columns=topics)
# df_NYT = pd.DataFrame(t_NYT, columns=topics)
# df_newsglobe = pd.DataFrame(t_newglobe, columns=topics)
print((t_breitbart.shape[0],t_newglobe.shape[0],t_NYT.shape[0]))
for iteration in range(max(t_breitbart.shape[0],t_newglobe.shape[0],t_NYT.shape[0])):
    plt.rc('font', **font)
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(18, 20), dpi=144, tight_layout=True)

    axs[0].set_title('Breitbart')
    axs[0].set_xlim([0, 1])
    axs[1].set_title('NYTimes')
    axs[1].set_xlim([0, 1])
    axs[2].set_title('Newsglobe24')
    axs[2].set_xlim([0, 1])
    if iteration >= t_breitbart.shape[0]:
        source1 = axs[0].barh(topics, t_breitbart[-1, :])
        axs[0].bar_label(source1)
    else:
        source1 = axs[0].barh(topics,t_breitbart[iteration,:])
        axs[0].bar_label(source1)

    if iteration >= t_NYT.shape[0]:
        source2 = axs[1].barh(topics, t_NYT[-1, :])
        axs[1].bar_label(source2)
    else:
        source2 = axs[1].barh(topics, t_NYT[iteration, :])
        axs[1].bar_label(source2)

    if iteration >= t_newglobe.shape[0]:
        source3 = axs[2].barh(topics, t_newglobe[-1, :])
        axs[2].bar_label(source3)
    else:
        source3 = axs[2].barh(topics, t_newglobe[iteration, :])
        axs[2].bar_label(source3)

    for ax in axs.flat:
        ax.set(xlabel='Trust value')
    fig.savefig('../imgs/img'+str(iteration) + '.png')
    plt.pause(0.5)

#     axs[0].bar(topics, t_breitbart[i,:])
#     plt.draw()
#     plt.show()
print(2)

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
    #     xt = np.arange(0, len(t)-49)
    #     axs[3].plot(xt, t[49::])
    # axs[3].set_title('Trust')
    # axs[3].set_ylim([0, 1])
    # axs[3].set_yticks(np.arange(0, 1, step=0.1))
    #
    # fig.tight_layout(pad=3, rect=[0, 0, 0.75, 1])
    # fig.suptitle(s, size=14, y=0.99)
    # plt.figure(figsize=(20, 20), dpi=80)
    # for ax in axs.flat:
    #     ax.set(xlabel='News requests during the time', ylabel='Metric value')
    #     ax.legend(sub_topics, loc='center left', bbox_to_anchor=(1, 0.5))
    #     # ax.legend(df['Topic'].unique(), loc='lower right')
    # plt.show()
