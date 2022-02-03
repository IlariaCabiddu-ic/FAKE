# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 15:11:10 2022

@author: luigi
"""

import requests
import time
import pandas as pd
from bs4 import BeautifulSoup as bs
"""
TOPICS:
tech,politics,entertainment,the-media,economy,sports

"""
class news:
    def __init__(self,title,link,topic):
        self.link = link
        self.title = title
        self.mainTopic = topic
        self.subtopics = list()
        self.text = ""
        self.retrieveText()
    def printNews(self):
        print("---Title: "+self.title+"\nLink: "+self.link+"---\n")
    def retrieveText(self):
        text = ""
        req = requests.get(self.link)
        content = bs(req.text,features="lxml")
        mainDiv = content.find('div',{'class':'entry-content'})
        try:
            for paragraph in mainDiv.findAll('p'):
                text += paragraph.text
                text += "\n"
            self.text = text
            footer = content.find('footer')
            topics = footer.find('p',{'class':'rmoreabt'})
            for t in topics.findAll('a'):
                self.subtopics.append(t.text)
        except:
            print("Error-"+self.link)
    def as_dict(self):
        return {'title':self.title,'text':self.text,'link':self.link,'mainTopic':self.mainTopic,'subtopics':str(self.subtopics)}
        
def getNewsList(link,topic, page):
    NewsList = list()
    req = requests.get(link+topic+"/page/"+str(page)+"/")
    content = bs(req.text,features="lxml")
    aList = content.find('section',{'class':'aList'})
    try:
        for article in aList.findAll('article'):
            l = article.find('a')
            NewsList.append(news(l.text,link[0:len(link)-1]+l['href'],topic))
            time.sleep(1)
    except:
        print("Error-"+link)
    return NewsList
def printNewsList(newsList):
    for n in newsList:
        n.printNews()
def createDF(newsList):
    df = pd.DataFrame(x.as_dict() for x in newsList)
    return df
    
if __name__ == "__main__":
    categoryList = ["tech","politics","entertainment","the-media","economy","sports"]
    MAX_PAGES=50
    NewsDF = pd.DataFrame(data=None,columns=['title', 'text', 'link', 'mainTopic', 'subtopics'])
    for t in categoryList:   
        for index in range(1,MAX_PAGES+1):
            newsList = getNewsList("https://www.breitbart.com/",t,index) #Importante mettere lo / finale nel link
            df = createDF(newsList)
            NewsDF = NewsDF.append(df)
        print(str((categoryList.index(t)+1)/len(categoryList)))
    NewsDF.to_csv('dataset.csv',index=None)