Some simple model like MetricF and SVD tested for recommendation.

you can  find SVD and MetricF model here.

1.. directory SVD_Recommendation include svd code.  
    you can easily and directly run the code by shell.  
  >python svd_recommendation.py  


2.. directory MetricF_Recommendation include MetricF model code.  
>1. add negative sample while train and test
>2. add both user bias and item bias and rate mean in model which is for eliminating for some useless rating. 
for example, some users always give full ratings to every movie default. or some users always give zero rating.  
>3. get topk item for user in test. using hit ratio adn NDCG evaluate model.  

download the movie_lens dataset and put to the input data_path exactly.(more explicit you can see the readme 
in MetricF_Recommendation)  

3.How to run

you can run the MetricF code by shell.  
>python main.py --data_path ../data/movie_lens_1m/ratings.csv        

4.Result  

set the negative ratio in train and test as 1:100.  
HR = 0.9250, NDCG = 0.6208 at last.     