# LFM-Recommendation
a simple LFM algorithm in recommendation

requires:
> torch cuda version >= 1.5 
> tensorboardX

env: 1050ti

first reproduce the model result show in paper(https://arxiv.org/abs/1802.04606) 
Loss result is about 0.7281, which is same as shown in paper.  
data split for train and test in 9:1

second recomposed the LFM model for a simple recommendation in dataset movie-length1M  
download the movie-lens and put the data in path  **data/movie_lens_1m/ratings.csv**
you can easily run the project by the shell command.
```shell script
python main.py --data_path ../data/movie_lens_1m/ratings.csv
```

For recommendation, we use the negative ratio and use HitRatio(HR) and NDCG as indicator.
also, you can find and change all the parameters in config.yaml 
set the negative ratio in train and test as 1:100, HR = 0.9250, NDCG = 0.6208 at last