# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 16:33:33 2022

@author: tkdgu
"""

if __name__ == '__main__':
    
    import os
    import sys
    import pickle
    import pandas as pd
    from collections import Counter
    
    directory = os.path.dirname(os.path.abspath(__file__))
    directory = directory.replace("\\", "/") # window
    os.chdir(directory)    
    
    sys.path.append(directory+'/submodule')

    directory = 'D:/아주대학교/tmlab - 문서/프로젝트/2021/아모레퍼시픽/데이터/제조/'
    # directory = 'D:/아주대학교/tmlab - 문서/프로젝트/2021/아모레퍼시픽/데이터/포장/'

    data = pd.DataFrame()
    
    for file in os.listdir(directory) :
        if '.csv' in file :
            temp_data = pd.read_csv(directory + file, skiprows=4)
            temp_data['db'] = file.split('_')[0]        
            data = pd.concat([data, temp_data]).reset_index(drop = 1)
        
    #%% 1. 데이터 전처리 
    
    # data_ = pd.DataFrame()
    
    # 전체 데이터 13만개 
    
    data = data[['번호','명칭','요약', '출원인대표명', '국제특허분류', '공통특허분류', '출원일','독립 청구항수', '전체 청구항수',
                 '대표 청구항','자국인용횟수', '자국피인용횟수' ,'INPADOC패밀리국가수','발명자수','최종 상태','db']]
    
    data.columns = ['pt_id', 'title', 'abstract', 'applicant', 'IPC', 'CPC', 'application_date', 'ind_claims_cnt', 'total_claims_cnt',
                    'claims_rep', 'forward_cites_cnt', 'backward_cites_cnt', 'family_country_cnt', 'inventor_cnt','state' ,'db']    

    data = data[['pt_id','title','abstract','CPC','application_date','claims_rep','state','db']]
    data = data.dropna(subset = ['application_date']).reset_index(drop = 1)

    data['application_year'] = data['application_date'].apply(lambda x : x.split('.')[0])
    data['TA'] = data['title'] +' '+ data['abstract']

    # before eda
    eda_df = pd.DataFrame()
    for db in sorted(set(data['db'])) :
        temp_data = data.loc[data['db'] == db, :]
        c = Counter(temp_data['application_year'])
        c = pd.DataFrame(c.values(), index = c.keys())
        c = c.sort_index()
        eda_df[db] = c
    
    
    #%% 2. 데이터 필터링
    
    from Levenshtein import distance as lev
    
    data_ = pd.DataFrame()
    
    for idx, row in data.iterrows() : 
        
        code_list = ['CN', 'US', 'EP', 'WO']
        
        if any(code in row['pt_id'] for code in code_list):
            
            state_list = ['Rejected', 'Withdrawn', 'Abandoned']
            
            if any(state == row['state'] for state in state_list):
                pass
            else :  
                if str(row['abstract']) != 'nan' :
                    data_ = data_.append(row)
    
    # 유사문서제거 
    for idx, row in data_.iterrows() :
        
        abstract = row['abstract'][0:100]
        
        for idx_, row_ in data_.iterrows() :
            if idx != idx_ :
                abstract_ = row_['abstract'][0:100]
                if lev(abstract, abstract_) <= 10 :
                    data_ = data_.drop(idx_)
                    print(idx_)
                    
    data_ = data_.reset_index(drop = 1)
    
    c = Counter(data_['db'])
    
    # 2. EDA AFTER
    
    data_['country'] = data_['pt_id'].apply(lambda x : x[0:2])
    country_list = data_['country']
    country_list = list(set(country_list))
    total_count = pd.DataFrame()
    
    for country in country_list :
        data_sample = data_.loc[data_['country'] == country,:]
        c = Counter(data_sample['application_year'])
        c = pd.DataFrame(c.values(), index = c.keys())
        c = c.sort_index()
        total_count[country] = c
    
    # 시계열 처리
    span_dict = {}
    span_dict['2015'] = 0
    span_dict['2016'] = 0
    span_dict['2017'] = 1
    span_dict['2018'] = 1
    span_dict['2019'] = 2
    span_dict['2020'] = 2
    span_dict['2021'] = 2
    
    data_['application_span'] = data_['application_year'].apply(lambda x : span_dict[x])
    
    total_count = pd.DataFrame()
    
    for country in country_list :
        data_sample = data_.loc[data_['country'] == country,:]
        c = Counter(data_sample['application_span'])
        c = pd.DataFrame(c.values(), index = c.keys())
        c = c.sort_index()
        total_count[country] = c
    
    #%% 3. 텍스트마이닝 및  lda 준비 
    
    # text preprocess
    from nltk.corpus import stopwords
    import spacy
    import re
    from collections import Counter
    import numpy as np 
    
    nlp = spacy.load("en_core_web_sm")
    nlp.enable_pipe("senter")
    stopwords_nltk = set(stopwords.words('english'))
    stopwords_spacy = nlp.Defaults.stop_words
    stopwords_add = ['method', 'invention', 'comprise', 'use', 'provide', 'present'
                     ,'relate', 'thereof', 'include', 'contain', 'disclose', 'effect']

    def preprocess_text(df, col) :
        
        # download('en_core_web_sm')
        # stopwords_.append('-PRON-')
        
        col_ = col + '_wordlist'
        df[col_] = [nlp(i) for i in df[col]]
        
        print('nlp finished')
        
        # get keyword
        df[col_] = [[token.lemma_.lower() for token in doc] for doc in df[col_]] # lemma
        df[col_] = [[token for token in doc if len(token) > 2] for doc in df[col_]] # 길이기반 제거
        df[col_] = [[re.sub(r"[^a-zA-Z0-9-]","",token) for token in doc ] for doc in df[col_]] #특수문자 교체    
        df[col_] = [[token for token in doc if not token.isdigit() ] for doc in df[col_]]  #숫자제거 
        df[col_] = [[token for token in doc if len(token) > 2] for doc in df[col_]] # 길이기반 제거
        df[col_] = [[token for token in doc if token not in stopwords_nltk] for doc in df[col_]] # 길이기반 제거
        df[col_] = [[token for token in doc if token not in stopwords_spacy] for doc in df[col_]] # 길이기반 제거
        df[col_] = [[token for token in doc if token not in stopwords_add] for doc in df[col_]] # 길이기반 제거
        
        return(df)
    
    # TF-IDF
    
    def tf_idf_counter(word_list) :
        temp = sum(word_list , [])
        c = Counter(temp)
        counter = pd.DataFrame(c.items())
        counter = counter.sort_values(1, ascending=False).reset_index(drop = 1)
        counter.columns = ['term', 'tf']
        counter = counter[counter['tf'] >= 3]
        counter['df'] = 0
        
        for idx,row in counter.iterrows() :
            term = row['term']
            for temp_list in word_list :
                if term in temp_list : counter['df'][idx] +=1
        
        counter['tf-idf'] = counter['tf'] / np.sqrt((1+ counter['df']))
        counter = counter.sort_values('tf-idf', ascending=False).reset_index(drop = 1)
        #counter = counter.loc[counter['tf-idf'] >= 1.5 , :].reset_index(drop = 1)
        return(counter)
    
    data_ = preprocess_text(data_, 'TA')
    
    #%% 4. TF-IDF, WC
    
    temp_data = data_.loc[data_['db'] == '보습제'].reset_index(drop = 1)
    
    # c = Counter(sum(data_['TA_wordlist'] , []))
    
    tf_idf = tf_idf_counter(temp_data['TA_wordlist'])
    tf_idf_times = pd.DataFrame()
    
    for time in [0,1,2] :
        data_sample = temp_data.loc[temp_data['application_span'] == time, :]
        temp = tf_idf_counter(data_sample['TA_wordlist'])
        temp.index = temp['term']
        temp = temp[['tf-idf']]
        tf_idf_times[time] = temp
        tf_idf['term'].tolist()
        
    from wordcloud import (WordCloud, get_single_color_func)
    import random
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    class GroupedColorFunc(object):
        """
        Uses different colors for different groups of words. 
        """
    
        def __init__(self, color_to_words, default_color):
            self.color_func_to_words = [
                (get_single_color_func(color), set(words))
                for (color, words) in color_to_words.items()]
    
            self.default_color_func = get_single_color_func(default_color)
    
        def get_color_func(self, word):
            """Returns a single_color_func associated with the word"""
            try:
                color_func = next(
                    color_func for (color_func, words) in self.color_func_to_words
                    if word in words)
            except StopIteration:
                color_func = self.default_color_func
    
            return color_func
    
        def __call__(self, word, **kwargs):
            return self.get_color_func(word)(word, **kwargs)
            return self.get_color_func(word)(word, **kwargs)
        
        # Define functions to select a hue of colors arounf: grey, red and green
    def red_color_func(word, font_size, position, orientation, random_state=None,
                        **kwargs):
        return "hsl(0, 100%%, %d%%)" % random.randint(30, 50)
    
    def green_color_func(word, font_size, position, orientation, random_state=None,
                        **kwargs):
        return "hsl(100, 100%%, %d%%)" % random.randint(20, 40)
    
    # tf_idf = pd.read_csv('./output/tf-idf_result.csv')

    # temp = tf_idf.loc[(tf_idf['sentiment_class'] == 'very positive')  | (tf_idf['sentiment_class'] == 'positive'), 'term'].tolist()
    # temp_ = tf_idf.loc[(tf_idf['sentiment_class'] == 'very negative')  | (tf_idf['sentiment_class'] == 'negative'), 'term'].tolist()

    # color_to_words = {
    #     # words below will be colored with a green single color function
    #     # '#00ff00': temp,
    #     'blue': temp,
    #     # will be colored with a red single color function
    #     'red': temp_
    # }

    # Words that are not in any of the color_to_words values
    # will be colored with a grey single color function
    # default_color = 'grey'
    # grouped_color_func = GroupedColorFunc(color_to_words, default_color)
    # data_sample = data_
    
    data_sample = temp_data
    tf_idf_ = tf_idf_counter(data_sample['TA_wordlist'])
    
    tf_idf_ = pd.merge(left = tf_idf_ , right = tf_idf, how = "left", on = "term", 
                       suffixes=('_sample', '_total')) 
    
    # tf_idf_ = tf_idf_.loc[(tf_idf_['sentiment_class'] == 'very negative')  | (tf_idf_['sentiment_class'] == 'negative'), :]
    # tf_idf_ = tf_idf_.loc[tf_idf_['sentiment_class'] == 'neutral', :]
    # tf_idf_ = tf_idf_.loc[(tf_idf_['sentiment_class'] == 'very positive')  | (tf_idf_['sentiment_class'] == 'positive'), :]
    
    dct = dict(zip(tf_idf_['term'], tf_idf_['tf-idf_sample']))
    mask_ = np.array(Image.open("./input/meta2.jpg"))
    
    
    wordcloud = WordCloud(
        background_color="white",
        mask = mask_
    )
    wordcloud = wordcloud.generate_from_frequencies(dct)
    
    # Apply our color function
    # wordcloud.recolor(color_func=grouped_color_func)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    
    #%% 5. LDA tunning
    
    from gensim.corpora import Dictionary
    import LDA_tunning
    import LDA_handling

    # with open('./output/LDA_result.pickle', 'rb') as f:
    #     LDA_obj = pickle.load(f)
    temp_data = data_.loc[data_['db'] == '클린'].reset_index(drop = 1)
    
    texts = temp_data['TA_wordlist']

    keyword_dct = Dictionary(texts)
    keyword_dct.filter_extremes(no_below = 3, no_above = 0.1)
    
    keyword_list = list(keyword_dct.token2id.keys())

    corpus = [keyword_dct.doc2bow(text) for text in texts]
    # encoded_keyword = embedding.keyword_embedding(keyword_list)
    texts = [[k for k in doc if k in keyword_list] for doc in texts]
    docs = [" ".join(i) for i in texts]
    
    tunning_result = LDA_tunning.tunning(texts, keyword_dct, corpus, 10, 31, 5)
    
    tunning_result.to_csv(directory + 'output/lda_tune_result_'+temp_data['db'][0]+'.csv',index = 0)
    

    # 6. LDA handling
    
    from sklearn.preprocessing import MinMaxScaler
    transformer = MinMaxScaler()
    temp = transformer.fit_transform(tunning_result[['Perplexity', 'U_mass']]) #MinMaxScaler 모델에 x_train_df 데이터 적용 (최소값, 최대값 계산)
    temp[:,0] = 1 - temp[:,0]
    # temp = temp[:,0] + temp[:,1] 
    temp = temp[:,0] + temp[:,1] - 0.02 * tunning_result['Topics']
    tunning_result['total'] = temp
    
    min_idx = tunning_result['total'].idxmax()
    
    LDA_obj = LDA_tunning.LDA_obj(texts, tunning_result['Topics'][min_idx]
                                  , tunning_result['Alpha'][min_idx], 
                                  tunning_result['Beta'][min_idx],
                                  keyword_dct) # best score 입력

    topic_doc_df = LDA_handling.get_topic_doc(LDA_obj.model, LDA_obj.corpus)
    topic_word_df = LDA_handling.get_topic_word_matrix(LDA_obj.model)
    topic_topword_df = LDA_handling.get_topic_topword_matrix(LDA_obj.model, 20)
    topic_time_df =  LDA_handling.get_topic_vol_time(LDA_obj.model, topic_doc_df, temp_data, 'application_span')
    
    topic_time_topn = {}
    topn_list = []
    for idx,row in topic_time_df.iterrows() :
        temp = row.tolist()
        topic_list = [temp.index(x) for x in sorted(temp, reverse = 1)]
        topic_time_topn[idx] = topic_list[0:5]
        topn_list.extend(topic_list[0:5])
        
    topn_list = list(set(topn_list))
    topic_time_topn_df = topic_time_df[topn_list]
    topic_time_topn_df.index = ['`15-`16', '`17-`18' , '`19-`21']
    
    topic_title_df = LDA_handling.get_most_similar_doc2topic(temp_data, topic_doc_df,5, 'title', 'application_year' )
    volumn_dict = LDA_handling.get_topic_vol(LDA_obj.model, LDA_obj.corpus)
    
    
    #결과저장
    import xlsxwriter  
    import pandas as pd
    # directory = 'C:/Users/tmlab/Desktop/작업공간/'
    writer = pd.ExcelWriter(directory+ 'output/LDA_results_'+temp_data['db'][0]+'2.xlsx', 
                            engine='xlsxwriter')
    
    topic_word_df.to_excel(writer , sheet_name = 'topic_word', index = 1)
    topic_topword_df.to_excel(writer , sheet_name = 'topic_topword', index = 1)
    topic_time_df.to_excel(writer , sheet_name = 'topic_time_vol', index = 1)
    topic_title_df.to_excel(writer , sheet_name = 'topic_doc_title', index = 1)
    pd.DataFrame(topic_doc_df).to_excel(writer , sheet_name = 'topic_doc', index = 1)
    temp = pd.DataFrame(topic_time_topn)
    temp.columns =  ['`15-`16', '`17-`18' , '`19-`21']
    temp.to_excel(writer , sheet_name = 'topic_time_topn', index = 1)
    topic_time_topn_df.to_excel(writer , sheet_name = 'topic_time_topn_vol', index = 1)
    
    topic_topn_topword_df = topic_topword_df[topic_time_topn[0]]
    try :topic_topn_topword_df = topic_topn_topword_df.join(topic_topword_df[topic_time_topn[1]])
    except : topic_topn_topword_df = pd.merge(topic_topn_topword_df, topic_topword_df[topic_time_topn[1]])
    try :topic_topn_topword_df = topic_topn_topword_df.join(topic_topword_df[topic_time_topn[2]])
    except : topic_topn_topword_df = pd.merge(topic_topn_topword_df, topic_topword_df[topic_time_topn[2]])
    
    topic_topn_topword_df.to_excel(writer , sheet_name = 'topic_topn_topword_vol', index = 1)
    

    writer.save()
    writer.close()
    
    