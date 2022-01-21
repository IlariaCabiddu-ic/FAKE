import pandas as pd
import model_trust
import matplotlib.pyplot as plt

df = [('P1', 'Politics', 0.7, 0.5, '2019-01-01 00:00:00'),
      ('E1', 'Economics', 0.4, 0.2, '2019-10-24 12:30:00'),
      ('A1', 'Actuality', 0.4, 0.1, '2017-04-12 03:32:00'),
      ('T1', 'Tech', 0.2, 0.4, '2018-05-20 05:10:33'),
      ('P2', 'Politics', 0.8, 0.8, '2020-01-01 10:00:00'),
      ('P1', 'Politics', 0.7, 0.5, '2019-01-01 00:00:00'),
      ('A2', 'Actuality', 0.2, 0.2, '2019-02-01 11:00:10'),
      ('E2', 'Economics', 0.2, 0.3, '2015-10-5 22:55:40'),
      ('A3', 'Actuality', 0.1, 0.3, '2010-03-22 00:00:00'),
      ('P3', 'Politics', 0.9, 0.8, '2020-01-01 03:10:00'),
      ('P4', 'Politics', 0.8, 0.6, '2022-01-03 14:30:40'),
      ('A4', 'Actuality', 0.1, 0.4, '2019-01-01 00:50:00'),
      ('P5', 'Politics', 0.7, 0.8, '2017-06-30 23:05:09'),
      ('A5', 'Actuality', 0.1, 0.3, '2018-09-06 01:55:03'),
      ('P6', 'Politics', 0.8, 0.5, '2016-11-06 13:44:44'),
      ('T2', 'Tech', 0.3, 0.2, '2012-02-22 22:22:22')]

# Create a DataFrame object
sub_df = pd.DataFrame(df, columns=['ID', 'Topic', 'Feedback', 'Message_based', 'Datetime'])
# print(sub_df)
topics = list(sub_df['Topic'].unique())
expertise = model_trust.compute_expertise(df=sub_df, topics=topics)
print("The expertise of analysed source is: ", expertise)
relevance = model_trust.compute_relevance(df=sub_df)
# print("The relevance of analysed source is: ", relevance)
goodwill = model_trust.compute_goodwill(df=sub_df, topics=topics, relevance=relevance)
print("The goodwill of analysed source is: ", goodwill)
coherence = model_trust.compute_coherence(df=sub_df, topics=topics)
print("The coherence of analysed source is: ", coherence)
trust = model_trust.compute_trust(expertise=expertise, goodwill=goodwill,
                                    coherence=coherence, topics=topics)
print("The trust of analysed source is: ", trust)
#
# plt.bar(expertise.keys(), expertise.values())
# plt.ylabel("Expertise value", fontsize="18")
# plt.xlabel("Topics", fontsize="18")
# plt.show()

# plt.bar(plt.bar(goodwill.keys(), goodwill.values()))
# plt.ylabel("Goodwill value", fontsize="18")
# plt.xlabel("Topics", fontsize="18")
# plt.show()
# 
# plt.bar(plt.bar(trust.keys(), trust.values()))
# plt.ylabel("Goodwill value", fontsize="18")
# plt.xlabel("Topics", fontsize="18")
# plt.show()
