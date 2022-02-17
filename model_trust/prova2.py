import model_trust
import warnings
import pandas as pd
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
""" There are several source news, the assessment is for each of them """

dataframe = [
             ('T11', 'source1', 'Tech', 0.7, 0.8, '2022-01-31 00:00:00'),
             ('T12', 'source1', 'Tech', 0.7, 0.7, '2022-01-11 10:00:00'),
             ('T13', 'source1', 'Tech', 0.8, 0.8, '2021-12-30 14:30:40'),
             ('T14', 'source1', 'Tech', 0.7, 0.7, '2021-12-20 23:05:09'),
             ('T15', 'source1', 'Tech', 0.7, 0.8, '2021-11-06 13:44:44'),
             ('T16', 'source1', 'Tech', 0.9, 0.8, '2021-10-01 01:10:55'),
             ('T17', 'source1', 'Tech', 0.7, 0.7, '2021-09-18 01:10:55'),
             ('T18', 'source1', 'Tech', 0.7, 0.7, '2021-08-20 01:10:55'),
             ('T19', 'source1', 'Tech', 0.8, 0.8, '2021-07-13 01:10:55'),
             ('T110', 'source1', 'Tech', 0.8, 0.8, '2021-07-03 01:10:55'),
             ('T111', 'source1', 'Tech', 0.8, 0.7, '2021-06-01 00:00:00'),
             ('T112', 'source1', 'Tech', 0.9, 0.9, '2021-05-11 10:00:00'),
             ('T113', 'source1', 'Tech', 0.8, 0.7, '2021-03-30 14:30:40'),
             ('T114', 'source1', 'Tech', 0.8, 0.8, '2021-02-20 23:05:09'),
             ('T115', 'source1', 'Tech', 0.8, 0.8, '2021-01-06 13:44:44'),
             ('T116', 'source1', 'Tech', 0.7, 0.7, '2020-10-17 01:10:55'),
             ('T117', 'source1', 'Tech', 0.9, 0.9, '2020-09-18 01:10:55'),
             ('T118', 'source1', 'Tech', 0.9, 0.9, '2020-08-20 01:10:55'),
             ('T119', 'source1', 'Tech', 0.8, 0.8, '2020-07-13 01:10:55'),
             ('T120', 'source1', 'Tech', 0.7, 0.8, '2020-05-16 01:10:55'),
             ('T121', 'source1', 'Tech', 0.8, 0.8, '2020-01-31 00:00:00'),
             ('T122', 'source1', 'Tech', 0.9, 0.8, '2020-01-11 10:00:00'),
             ('T123', 'source1', 'Tech', 0.7, 0.8, '2019-12-30 14:30:40'),
             ('T124', 'source1', 'Tech', 0.9, 0.9, '2019-12-20 23:05:09'),
             ('T125', 'source1', 'Tech', 0.8, 0.9, '2019-11-06 13:44:44'),
             ('T126', 'source1', 'Tech', 0.9, 0.9, '2019-10-01 01:10:55'),
             ('T127', 'source1', 'Tech', 0.7, 0.7, '2019-09-18 01:10:55'),
             ('T128', 'source1', 'Tech', 0.8, 0.7, '2019-08-20 01:10:55'),
             ('T129', 'source1', 'Tech', 0.8, 0.8, '2019-07-13 01:10:55'),
             ('T130', 'source1', 'Tech', 0.9, 0.9, '2019-07-03 01:10:55'),
             ('T131', 'source1', 'Tech', 0.7, 0.7, '2019-06-01 00:00:00'),
             ('T132', 'source1', 'Tech', 0.9, 0.9, '2019-05-11 10:00:00'),
             ('T133', 'source1', 'Tech', 0.8, 0.8, '2019-03-30 14:30:40'),
             ('T134', 'source1', 'Tech', 0.7, 0.8, '2019-02-20 23:05:09'),
             ('T135', 'source1', 'Tech', 0.8, 0.7, '2019-01-06 13:44:44'),
             ('T136', 'source1', 'Tech', 0.9, 0.8, '2018-10-17 01:10:55'),
             ('T137', 'source1', 'Tech', 0.8, 0.8, '2018-09-18 01:10:55'),
             ('T138', 'source1', 'Tech', 0.9, 0.9, '2018-08-20 01:10:55'),
             ('T139', 'source1', 'Tech', 0.8, 0.8, '2018-07-13 01:10:55'),
             ('T140', 'source1', 'Tech', 0.8, 0.8, '2018-05-16 01:10:55'),
             # ----------------------another topic--------------------------
             ('A11', 'source1', 'Actuality', 0.8, 0.7, '2019-12-12 03:32:00'),
             # ('A12', 'source1', 'Actuality', 0.7, 0.7, '2019-11-01 11:00:10'),
             # ('A13', 'source1', 'Actuality', 0.9, 0.9, '2018-03-22 00:00:00'),
             # ('A16', 'source1', 'Actuality', 0.5, 0.7, '2017-11-12 23:32:00'),
             # ('A16', 'source1', 'Actuality', 0.8, 0.8, '2017-01-12 23:32:00'),

             # ----------------------ANOTHER SOURCE--------------------------
             ('T21', 'source2', 'Tech', 0.8, 0.8, '2022-01-31 00:00:00'),
             # ('T22', 'source2', 'Tech', 0.8, 0.7, '2022-01-11 10:00:00'),
             # ('T23', 'source2', 'Tech', 0.7, 0.7, '2021-12-30 14:30:40'),
             # ('T24', 'source2', 'Tech', 0.7, 0.8, '2021-12-20 23:05:09'),
             # ('T25', 'source2', 'Tech', 0.8, 0.8, '2021-11-06 13:44:44'),
             # ('T26', 'source2', 'Tech', 0.8, 0.8, '2021-10-01 01:10:55'),
             # ('T27', 'source2', 'Tech', 0.7, 0.7, '2021-09-18 01:10:55'),
             # ('T28', 'source2', 'Tech', 0.8, 0.7, '2021-08-20 01:10:55'),
             # ('T29', 'source2', 'Tech', 0.8, 0.8, '2021-07-13 01:10:55'),
             # ('T210', 'source2', 'Tech', 0.7, 0.8, '2021-07-03 01:10:55'),
             # # ----------------------another topic--------------------------
             # ('E21', 'source2', 'Economics', 0.4, 0.2, '2019-10-24 12:30:00'),
             # ('P21', 'source2', 'Politics', -0.2, 0.1, '2018-05-20 05:10:33'),
             # ('E22', 'source2', 'Economics', 0.2, 0.3, '2015-10-5 22:55:40'),
             ('A24', 'source2', 'Actuality', -0.8, -0.8, '2019-01-01 00:50:00')]
             # ('A25', 'source2', 'Actuality', 0.1, 0.3, '2018-09-06 01:55:03'),
             # ('P22', 'source2', 'Politics', -0.3, 0.1, '2012-02-22 22:22:22'),
             # ('E23', 'source2', 'Economics', 0.4, 0.4, '2019-12-24 15:30:00')]


# Create a DataFrame object
df = pd.DataFrame(dataframe, columns=['ID', 'Source', 'Topic', 'Feedback', 'Message_based', 'Datetime'])
# print(df)
"-----------------analysis of each present source--------------"

sources = list(df['Source'].unique())
for s in sources:
    sub_df = df[df['Source'] == s]  # iteration for the each topic
    topics = list(sub_df['Topic'].unique())
    expertise = model_trust.compute_expertise(df=sub_df, topics=topics)
    print("The expertise of analysed source '", s, "' is: ", expertise)

    relevance = model_trust.compute_relevance(df=sub_df, topics=topics)
    print("The relevance of analysed source '", s, "' is: ", relevance)
    goodwill = model_trust.compute_goodwill(df=sub_df, topics=topics, relevance=relevance)
    print("The goodwill of analysed source '", s, "' is: ", goodwill)
    coherence = model_trust.compute_coherence(df=sub_df, topics=topics)
    print("The coherence of analysed source '", s, "' is: ", coherence)
    trust = model_trust.compute_trust(expertise=expertise, goodwill=goodwill,
                                      coherence=coherence, topics=topics)
    print("The trust of analysed source '", s, "' is: ", trust)
