import pandas as pd
import model_trust

df = [('Economics', 34, 7, 5),
      ('Politics', 31, 5, 7),
      ('Sport', 16, 5, 1),
      ('Tech', 31, 7, 7),
      ('Economics', 31, 10, 4),
      ('Sport', 35, 2, 5),
      ('Politics', 35, 15, 11),
      ('Sport', 16, 5, 2),
      ('Economics', 40, 13, 7),
      ('Economics', 31, 6, 4),
      ('Sport', 35, 3, 5),
      ('Sport', 16, 6, 11),
      ('Sport', 31, 18, 7),
      ('Economics', 31, 4, 4),
      ('Tech', 35, 10, 5)]
# Create a DataFrame object
sub_df = pd.DataFrame(df, columns=['Topic', 'n_words', 'Unique_word', 'Typos'])
topics = list(sub_df['Topic'].unique())
expertise = model_trust.compute_expertise(df=sub_df, topics=topics)
print(expertise)
