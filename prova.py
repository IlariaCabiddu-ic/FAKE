import pandas as pd
import model_trust

df = [('E1', 'Economics', 0.7, 34, 7, 5),
      ('P1', 'Politics', 0.4, 31, 6, 2),
      ('S1', 'Sport', -0.4, 16, 1, 5),
      ('T1', 'Tech', 0.2, 31, 7, 7),
      ('E1', 'Economics', 0.7, 34, 7, 5),
      ('S2', 'Sport', -0.2, 35, 2, 5),
      ('P2', 'Politics', 0.2, 35, 15, 11),
      ('S3', 'Sport', 0.1, 16, 5, 2),
      ('E3', 'Economics', 0.6, 40, 13, 7),
      ('E4', 'Economics', 0.4, 31, 6, 4),
      ('S4', 'Sport', -0.5, 35, 3, 5),
      ('S5', 'Sport', -0.7, 16, 6, 11),
      ('S6', 'Sport', 0.2, 31, 18, 7),
      ('E5', 'Economics', 0.5, 31, 4, 4),
      ('T2', 'Tech', 0.3, 35, 10, 5)]
# Create a DataFrame object
sub_df = pd.DataFrame(df, columns=['ID', 'Topic', 'Feedback', 'n_words', 'Unique_word', 'Typos'])
topics = list(sub_df['Topic'].unique())
expertise = model_trust.compute_expertise(df=sub_df, topics=topics)
print("The expertise of analysed source is: ", expertise)
relevance = model_trust.compute_relevance(df=sub_df)
# print(relevance)
goodwill = model_trust.compute_goodwill(df=sub_df, topics=topics, relevance=relevance)
print("The goodwill of analysed source is: ", goodwill)
