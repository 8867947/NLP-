# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 09:30:28 2021

@author: Priya
"""

import nltk

para = """Wikipedia was founded as an offshoot of Nupedia, a now-abandoned project to produce a free encyclopedia, begun by the online media company Bomis. Nupedia had an elaborate system of peer review and required highly qualified contributors, but articles' writing was slow. During 2000, Jimmy Wales (founder of Nupedia and co-founder of Bomis), and Larry Sanger, whom Wales had employed to work on the encyclopedia project, discussed ways of supplementing Nupedia with a more open, complementary project. Multiple sources suggested that a wiki might allow public members to contribute material, and Nupedia's first wiki went online on January 10, 2001.

There was considerable resistance on the part of Nupedia's editors and reviewers to the idea of associating Nupedia with a website in the Wiki format, so the new project was given the name "Wikipedia" and launched on its own domain, wikipedia.com, on January 15 (now called "Wikipedia Day" by some users). The bandwidth and server (in San Diego) were donated by Wales. Other current and past Bomis employees who have worked on the project include Tim Shell, one of the cofounders of Bomis and its current CEO, and programmer Jason Richey.

In May 2001, a large number of non-English Wikipedias were launched—in Catalan, Chinese, Dutch, Esperanto, French, German, Hebrew, Italian, Japanese, Portuguese, Russian, Spanish, and Swedish. These were soon joined by Arabic and Hungarian. In September,[2] Polish was added, and further commitment to the multilingual provision of Wikipedia was made. At the end of the year, Afrikaans, Norwegian, and Serbo-Croatian versions were announced.

The domain was eventually changed to the present wikipedia.org when the Wikimedia Foundation was launched, in 2003, as its new parent organization with the ".org" top-level domain denoting its not-for-profit nature. Today, there are Wikipedias in more than 300 languages

"""

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

sentences = nltk.sent_tokenize(para)
ps = PorterStemmer()
wordnet = WordNetLemmatizer()
data_list=[]

for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    data_list.append(review)
    
    
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(data_list).toarray()