#!/usr/bin/env python
# -*- coding: utf-8 -*-


import io
import json
import nltk
import pymorphy2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

morph = pymorphy2.MorphAnalyzer()
with io.open('ml_contest/train.json','r',encoding="utf-8") as data_file:
	#print "opening file",i_file
	news = json.load(data_file)
	news_collection = []
	sentiment_collection = []
	print len(news)
	for idx,item in enumerate(news):
	    news_tokens = ""
	    #print idx
	    #print item['text'][:30]
	    #print item['sentiment']
	    text = item['text']
	    sentiment_collection.append(item['sentiment'])
	    tokens = [i.lower() for i in nltk.word_tokenize(text)]
	    for token in tokens:
		parseds = morph.parse(token)
		#news_tokens = news_tokens+" "+token
		news_tokens=news_tokens+" "+parseds[0].normal_form
	    	#print token.encode('utf8')," :",parseds[0].normal_form
		'''for parsed in parseds:
			print parsed.normal_form,
		print'''
	    news_collection.append(news_tokens)
	    #titles.append(strip_tags(item['title']))
	    #descriptions.append(strip_tags(item['fulltext']))
	    #descriptions.append(strip_tags(item['title']))
	    #print idx,item['title'].encode('utf8')
	    if idx == 2000:
	        break
	print "finishing"
	#print news_collection
	count_vect = CountVectorizer()
	X_train_counts = count_vect.fit_transform(news_collection)
	#print X_train_counts.shape
	#print count_vect.vocabulary_.get(u'год')
	#print X_train_counts[:,273]

	tf_transformer = TfidfTransformer()
	X_train_tfidf = tf_transformer.fit_transform(X_train_counts)
	#print X_train_tfidf.shape

	docs_new = []
	categories = []
	for i in range(7000,7100):
	    docs_new.append(news[i]['text'])
	    categories.append(news[i]['sentiment'])

	X_new_counts = count_vect.transform(docs_new)
	X_new_tfidf = tf_transformer.transform(X_new_counts)

	clf = MultinomialNB().fit(X_train_tfidf,sentiment_collection)
	predicted = clf.predict(X_new_tfidf)

	for doc,predicted_category,actual_category in zip(docs_new, predicted,categories):
	    #print ('%r => %s' % (doc.encode("UTF-8"),category))
	    if actual_category=='negative':
	    	print predicted_category,actual_category,"=>",doc.encode('utf8')[:100]
	print predicted_category
	print actual_category
	print confusion_matrix(categories,predicted,labels=["positive","negative","neutral"])
