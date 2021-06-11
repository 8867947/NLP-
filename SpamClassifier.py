import pandas as pd
para = pd.read_csv("F:\\MTech\\SelfStudies\\NLP\\smsspamcollection\\SMSSpamCollection", sep='\t',names=['target','label'])
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

ps= PorterStemmer()
wordnet = WordNetLemmatizer()
data_list=[]
for i in range(0, len(para)):
    data = re.sub('[^a-zA-Z]', ' ', para['label'][i])
    data = data.lower()
    data = data.split()
    data = [ps.stem(word) for word in data if not word in set(stopwords.words('english'))]
    data= ' '.join(data)
    data_list.append(data)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
X= cv.fit_transform(data_list).toarray()

y = pd.get_dummies(para['target'])
y = y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state= 0)

from sklearn.naive_bayes import MultinomialNB
spam_detect = MultinomialNB().fit(X_train,y_train)

y_pred = spam_detect.predict(X_test)

from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print(ac)