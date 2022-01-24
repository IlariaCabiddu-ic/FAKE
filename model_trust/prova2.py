import model_trust
import warnings
import pandas as pd
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
""" There are several source news, the assessment is for each of them """

dataframe = [('P1', 'source1', 'Politics', 0.7, 0.5, '2019-01-01 00:00:00'),
      ('E1', 'source2', 'Economics', 0.4, 0.2, '2019-10-24 12:30:00'),
      ('A1', 'source1', 'Actuality', 0.4, 0.1, '2017-04-12 03:32:00'),
      ('T1', 'source2', 'Tech', -0.2, 0.1, '2018-05-20 05:10:33'),
      ('P2', 'source1', 'Politics', 0.8, 0.8, '2020-01-01 10:00:00'),
      ('P1', 'source1', 'Politics', 0.7, 0.5, '2019-01-01 00:00:00'),
      ('A2', 'source1', 'Actuality', 0.2, 0.2, '2019-02-01 11:00:10'),
      ('E2', 'source2', 'Economics', 0.2, 0.3, '2015-10-5 22:55:40'),
      ('A3', 'source1', 'Actuality', 0.1, 0.3, '2010-03-22 00:00:00'),
      ('P3', 'source2', 'Politics', 0.9, 0.8, '2020-01-01 03:10:00'),
      ('P4', 'source1', 'Politics', 0.8, 0.6, '2022-01-03 14:30:40'),
      ('A4', 'source2', 'Actuality', 0.1, 0.4, '2019-01-01 00:50:00'),
      ('P5', 'source1', 'Politics', 0.7, 0.8, '2017-06-30 23:05:09'),
      ('A5', 'source2', 'Actuality', 0.1, 0.3, '2018-09-06 01:55:03'),
      ('P6', 'source1', 'Politics', 0.8, 0.5, '2016-11-06 13:44:44'),
      ('T2', 'source2', 'Tech', -0.3, 0.1, '2012-02-22 22:22:22'),
      ('P7', 'source1', 'Politics', 0.7, 0.5, '2017-03-01 01:10:55'),
      ('E3', 'source2', 'Economics', 0.4, 0.4, '2019-12-24 15:30:00'),
      ('A6', 'source1', 'Actuality', 0.3, 0.2, '2019-11-12 23:32:00'),
      ('T3', 'source2', 'Tech', -0.5, 0.1, '2021-05-30 15:10:33'),
      ('P8', 'source1', 'Politics', 0.8, 0.7, '2022-12-01 20:00:00'),
      ('P9', 'source2', 'Politics', 0.9, 0.9, '2019-12-31 11:10:30'),
      ('A7', 'source1', 'Actuality', 0.3, 0.3, '2019-07-01 19:00:10'),
      ('E4', 'source2', 'Economics', 0.2, 0.4, '2014-10-5 22:55:40'),
      ('A8', 'source1', 'Actuality', 0.1, 0.1, '2018-09-22 20:00:00'),
      ('P10', 'source2', 'Politics', 0.9, 0.8, '2017-11-01 11:10:00'),
      ('P11', 'source1', 'Politics', 0.8, 0.7, '2020-01-03 15:30:40'),
      ('A9', 'source2', 'Actuality', 0.2, 0.4, '2021-01-01 10:50:00'),
      ('P12', 'source1', 'Politics', 0.7, 0.7, '2020-08-30 23:05:09'),
      ('A10', 'source2', 'Actuality', 0.2, 0.3, '2021-09-06 18:55:03'),
      ('P13', 'source1', 'Politics', 0.8, 0.9, '2017-11-06 17:44:44'),
      ('T4', 'source1', 'Tech', -0.3, 0.1, '2018-02-22 23:23:23'),
      ('P14', 'source3', 'Politics', 0.8, 0.9, '2022-01-01 00:00:00')
      ]

# Create a DataFrame object
df = pd.DataFrame(dataframe, columns=['ID', 'Source', 'Topic', 'Feedback', 'Message_based', 'Datetime'])
"-----------------analyzing of each present source--------------"

sources = list(df['Source'].unique())
for s in sources:
    sub_df = df[df['Source'] == s]  # iteration for the each topic
    topics = list(sub_df['Topic'].unique())
    expertise = model_trust.compute_expertise(df=sub_df, topics=topics)
    # print("The expertise of analysed source '", s, "' is: ", expertise)
    relevance = model_trust.compute_relevance(df=sub_df)
    # print("The relevance of analysed source '", s, "' is: ", relevance)
    goodwill = model_trust.compute_goodwill(df=sub_df, topics=topics, relevance=relevance)
    # print("The goodwill of analysed source '", s, "' is: ", goodwill)
    coherence = model_trust.compute_coherence(df=sub_df, topics=topics)
    # print("The coherence of analysed source '", s, "' is: ", coherence)
    trust = model_trust.compute_trust(expertise=expertise, goodwill=goodwill,
                                      coherence=coherence, topics=topics)
    print("The trust of analysed source '", s, "' is: ", trust)
