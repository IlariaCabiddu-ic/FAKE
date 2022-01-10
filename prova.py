import pandas as pd
import model_trust

df = [('E1', 'Economics', 0.7, 0.5, '2019-01-01 00:00:00'),
      ('P1', 'Politics', 0.4, 0.2, '2019-10-24 12:30:00'),
      ('S1', 'Sport', -0.4, 0.1, '2017-04-12 03:32:00'),
      ('T1', 'Tech', 0.2, 0.4, '2018-05-20 05:10:33'),
      ('E1', 'Economics', 0.7, 0.5, '2019-01-01 00:00:00'),
      ('S2', 'Sport', -0.2, 0.2, '2019-02-01 11:00:10'),
      ('P2', 'Politics', 0.2, 0.3, '2015-10-5 22:55:40'),
      ('S3', 'Sport', 0.1, 0.3, '2010-03-22 00:00:00'),
      ('E3', 'Economics', 0.6, 0.8, '2020-01-01 03:10:00'),
      ('E4', 'Economics', 0.4, 0.6, '2022-01-03 14:30:40'),
      ('S4', 'Sport', -0.5, 0.4, '2019-01-01 00:50:00'),
      ('S5', 'Sport', -0.7, 0.3, '2018-09-06 01:55:03'),
      ('S6', 'Sport', 0.2, 0.2, '2017-06-30 23:05:09'),
      ('E5', 'Economics', 0.5, 0.5, '2016-11-06 13:44:44'),
      ('T2', 'Tech', 0.3, 0.2, '2012-02-22 22:22:22')]
# Create a DataFrame object
sub_df = pd.DataFrame(df, columns=['ID', 'Topic', 'Feedback', 'Message_based', 'Datetime'])
topics = list(sub_df['Topic'].unique())
expertise = model_trust.compute_expertise(df=sub_df, topics=topics)
# print("The expertise of analysed source is: ", expertise)
relevance = model_trust.compute_relevance(df=sub_df)
print("The relevance of analysed source is: ", relevance)
goodwill = model_trust.compute_goodwill(df=sub_df, topics=topics, relevance=relevance)
# print("The goodwill of analysed source is: ", goodwill)
historical = model_trust.compute_historical(df=sub_df, topics=topics)
print("The historical of analysed source is: ", historical)
trust = model_trust.compute_trust(expertise=expertise, goodwill=goodwill,
                                  historical=historical,topics=topics)
