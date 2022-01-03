import pandas as pd

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
sub_df= df[df["Source"] == s]
# detail sub_df
sub_df.info()

