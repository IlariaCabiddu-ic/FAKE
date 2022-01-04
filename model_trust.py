import collections

def compute_expertise(sub_df, topics):
    alfa = 0.5
    beta = 0.5
    M_st = []
    Q_st = []
    expertise = []
    '''computation of M_st that is a percentage number of published news items 
        of a specific topic compared to the total number of news items of a source'''
    M_st = sub_df['Topic'].to_list()
    M_st = collections.Counter(M_st)
    sum_p = sum(M_st.values())
    for t in range(len(M_st)):
        M_st[t] = float(M_st[t]/sum_p)

    expertise = alfa * M_st + beta * Q_st

    '''import pandas as pd
import collections

empoyees = [('jack', 34, 'Sydney', 5) ,
         ('Riti', 31, 'Delhi' , 7) ,
         ('Aadi', 16, 'Sydney', 11) ,
         ('Mohit', 31,'Delhi' , 7) ,
         ('Veena', 31, 'Delhi' , 4) ,
         ('Shaunak', 35, 'Mumbai', 5 ),
         ('Shaun', 35, 'Colombo', 11)
          ]
# Create a DataFrame object
sub_df = pd.DataFrame(empoyees, columns=['Name', 'Age', 'City', 'Experience'])
sub_sub = sub_df['City'].to_list()

frequency_topic=dict(collections.Counter(sub_sub))
print(frequency_topic)
sum_p = sum(frequency_topic.values())
print(sum_p)
for t in frequency_topic:
    frequency_topic[t]=float(frequency_topic[t]/sum_p)
print(frequency_topic)'''