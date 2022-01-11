'''--------------Main file-------------- '''
import pandas as pd
import model_trust


df_new = pd.read_json('path_of_the_file.json') # if we have a online repository, we can use the URL, df= pd.read_json(URL)
# visualization of info dataframe (optional)
df_new.info()

# update of dataframe (eventually there are iterations during the time)
df = pd.DataFrame()
df.append(df_new)

# STEP ONE  filtering of the specific source
print("Enter the name of the source to be analysed:\n ")
s = input()
print("The source to be analysed is " + s)

# create a df with only data of source s
sub_df = df[df["Source"] == s]
# detail sub_df
sub_df.info()

# STEP TWO check Top Level Domain (TLD)
i = len(s)-1
TLD = ''
while s[i] != '.':
    TLD = TLD + s[i]
    i = i-1
TLD = TLD[::-1]  # TLD extraction of the source to be analysed
# print(TLD)

TLDs = ['gov', 'edu', 'mil', 'museum', 'jobs', 'post', 'travel']

if TLD in TLDs:
    print("This source is trusted!!!\n")
else:
    print("This source needs to be analysed\n")
    # STEP THREE Detection of all topic (optional maybe)
    topics = sub_df['Topic'].unique()

    """--------------- MODEL TRUST ---------------------"""
    "Computation of the source expertise"
    expertise = model_trust.compute_expertise(df=sub_df, topics=topics)
    print(expertise)
    "Computation of the news relevance"
    relevance = model_trust.compute_relevance(df=sub_df)
    print(relevance)
    "Computation of the source goodwill"
    goodwill = model_trust.compute_goodwill(df=sub_df, topics=topics, relevance=relevance)
    print("The goodwill of analysed source is: ", goodwill)
    historical = model_trust.compute_historical(df=sub_df, topics=topics)
    # print("The historical of analysed source is: ", historical)
    trust = model_trust.compute_trust(expertise=expertise, goodwill=goodwill,
                                      historical=historical, topics=topics)
    print("The trust of analysed source is: ", trust)
    
