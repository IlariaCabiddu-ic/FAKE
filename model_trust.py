import collections


def compute_expertise(df, topics):
    alfa = 0.5
    beta = 0.5
    Q_st = []
    q = []
    dictionary_Q = {}
    dictionary_E = {}

    '''computation of M_st that is a percentage number of published news items 
        of a specific topic compared to the total number of news items of a source'''
    df = df.drop_duplicates()
    "Compute focus theme M_st "
    M_st = df['Topic'].to_list()
    M_st = dict(collections.Counter(M_st))
    # print(M_st)
    "Number of total news"
    N_s = sum(M_st.values())
    # print(N_s)
    for i in M_st:
        M_st[i] = float(round(M_st[i]/N_s, 4))
    # print(M_st)

    "Compute technicality Q_st"
    for t in topics:
        df_sub = df[df['Topic'] == t]  # iteration for the each topic
        # print(df_sub)
        for j in range(df_sub.shape[0]):
            # print((df_sub['Unique_word']- df_sub['Typos'])/ df_sub['n_words'])
            q = sum((df_sub['Unique_word'] - df_sub['Typos']) / df_sub['n_words'])
        # print(q)
        # q = sum(q)
        # print(df_sub.shape[0])
        Q_st.append(q / (df_sub.shape[0]))  # divide by number of topic news P_st

        dictionary_Q[t] = q / (df_sub.shape[0])
    # print(Q_st)
    # print(dictionary_Q)
    zipped_lists = zip(list(M_st.values()), Q_st)
    expertise = [alfa * x + beta*y for (x, y) in zipped_lists]
    expertise = [round(x, 2) for x in expertise]
    for t in range(len(topics)):
        dictionary_E[topics[t]] = expertise[t]
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
    # print(df_unique)
    for t in topics:
        df_sub = df_unique[df_unique['Topic'] == t]  # iteration for the each topic
        # print(df_sub)
        for j in range(df_sub.shape[0]):
            g = round(sum(df_sub['Relevance'] * df_sub['Feedback']), 4)
        # print(df_sub.shape[0], g)
        G_st.append(g / (df_sub.shape[0]))  # divide by number of topic news P_st
        dictionary_G[t] = g / (df_sub.shape[0])
    # print(dictionary_G)
    return dictionary_G
