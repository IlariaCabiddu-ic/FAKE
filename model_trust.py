import collections


def compute_expertise(df, topics):
    alfa = 0.5
    beta = 0.5
    Q_st = []
    q = []
    dictionary_Q = {}

    '''computation of M_st that is a percentage number of published news items 
        of a specific topic compared to the total number of news items of a source'''

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
        df_sub = df[df['Topic'] == t]
        # print(df_sub)
        for j in range(df_sub.shape[0]):
            # print((df_sub['Unique_word']- df_sub['Typos'])/ df_sub['n_words'])
            q = sum((df_sub['Unique_word'] - df_sub['Typos']) / df_sub['n_words'])
            # print(q)
        # q = sum(q)
        Q_st.append(q)
        dictionary_Q[t] = q
    # print(dictionary_Q)
    zipped_lists = zip(list(M_st.values()), Q_st)
    expertise = [alfa*x + beta*y for (x, y) in zipped_lists]
    expertise = [round(x, 2) for x in expertise]
    return expertise
