# %% [markdown]
# # Trying out the Surpr!se Library

# %%
from IPython.display import display, HTML

display(HTML(data="""
<style>
    div#notebook-container    { width: 95%; }
    div#menubar-container     { width: 65%; }
    div#maintoolbar-container { width: 99%; }
</style>
"""))

# %%
# !{sys.executable} -m pip install --upgrade scikit-learn
# !{sys.executable} -m pip install --upgrade scipy
# !{sys.executable} -m pip install --upgrade  seaborn
# !{sys.executable} -m pip install --upgrade  uncertainties
# !{sys.executable} -m pip install surprise

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import scipy
from scipy import sparse

import sklearn
from sklearn.metrics import ndcg_score

import surprise
from surprise import Dataset, accuracy, NormalPredictor, BaselineOnly, KNNBasic, KNNWithMeans, KNNWithZScore, SlopeOne,  CoClustering,  SVD, NMF, SVDpp
from surprise.model_selection import cross_validate, KFold, RepeatedKFold,  train_test_split

import copy
import time
import sys
from itertools import starmap, product
import multiprocessing

import statistics
import uncertainties
from uncertainties import ufloat 


# %%
print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('The scipy version is {}.'.format(scipy.__version__))
print('The seaborn version is {}.'.format(sns.__version__))
print('The uncertainties version is {}.'.format(uncertainties.__version__))
print('The surprise version is {}.'.format(surprise.__version__))


# %%
# the URL for jester in surprise is wrong. 
# you have to manually correct it in the installed package, in {environment}/lib/python3.6/site-packages/surprise/builtin_datasets.py
# CORRECT: http://eigentaste.berkeley.edu/dataset/archive/jester_dataset_2.zip  
# jester = Dataset.load_builtin('jester') 

# %% [markdown]
# ## Dataset manipulation functions

# %%
def limit_dataset_number(dataset_in, number):
    
    dataset = copy.deepcopy(dataset_in)
    
    dataset.raw_ratings = dataset.raw_ratings[0:number]
    return dataset


# %%
def limit_dataset_to_users_w_between_ratings(dataset_in, lowerlimit, upperlimit, verbose=False):
    
    dataset = copy.deepcopy(dataset_in)
    
    new_raw_ratings = []
    
    curr_user='0'
    curr_user_ratings = []
    for rating in dataset.raw_ratings:
        
        if(curr_user != rating[0]):
            
            if( ( len(curr_user_ratings) >= lowerlimit ) and ( len(curr_user_ratings) <= upperlimit  ) ):
                new_raw_ratings.extend(curr_user_ratings)
                
            curr_user = rating[0]
            curr_user_ratings = []
                
        curr_user_ratings.append(rating)

    if verbose: print("dataset reduced from %s to %s ratings" % ( len(dataset.raw_ratings), len(new_raw_ratings) ))
        
    dataset.raw_ratings = new_raw_ratings
    
    return dataset


# %%
# jester_tiny = limit_dataset_to_users_w_between_ratings(jester, 3, 5, verbose=False)

# %%
# reindexed_jester_tiny  = reindex_dataset(jester_tiny)

# %%
def rescale_dataset(dataset_in):
    
    dataset = copy.deepcopy(dataset_in)
    
    new_raw_ratings = []
    for rating in dataset.raw_ratings:
        
        rescaled_rating_2 = round(0.2*rating[2]+3, 1)
        
        new_raw_ratings.append(   (rating[0], rating[1], rescaled_rating_2 ,  rating[3])   )
        
    dataset.raw_ratings = new_raw_ratings
    dataset.reader.rating_scale=(1, 5)
    
    return dataset


# %%
def dataset_info(dataset, verbose=False):

    trainset = dataset.build_full_trainset()

    print("n_users = %s" % trainset.n_users)
    print("average number of ratings per user = %.1f " % (trainset.n_ratings / trainset.n_users) )
    
    if verbose: 
        print("n_ratings = %s" % trainset.n_ratings)
        print("n_items = %s" % trainset.n_items)
        print("rating_scale = %s" +  str(trainset.rating_scale) )
        print(dataset.raw_ratings[0:5])


# %%
def dataset_desc_str(dataset):
    trainset = dataset.build_full_trainset()
    
    ratings_p_user = int(trainset.n_ratings / trainset.n_users)
    n_ratings = trainset.n_ratings
    
    return( "_%sRat_%sTot" % (ratings_p_user, n_ratings ) )


# %%
def reindex_dataset(dataset_in, exclude_time=True):
    dataset = copy.deepcopy(dataset_in)
    
    for rating in dataset.raw_ratings:  
        rating = (int(rating[0]), int(rating[1]), rating[2], rating[3] ) 
    
    def get_repl_dict(arr): 
        return  { value : index for (value, index) in zip( sorted(list(set(arr)))  , range(0,len(arr))   )  }   
    
    repl_dict_0 = get_repl_dict([int(p[0]) for p in dataset.raw_ratings ])
    repl_dict_1 = get_repl_dict([int(p[1]) for p in dataset.raw_ratings ])
    
    new_raw_ratings = []
    
    for rating in dataset.raw_ratings:
        
        if exclude_time:
            new_rating = (repl_dict_0[int(rating[0])], repl_dict_1[int(rating[1])], rating[2],None) 
        else:
            new_rating = (repl_dict_0[int(rating[0])], repl_dict_1[int(rating[1])], rating[2], rating[3] ) 
            
        new_raw_ratings.append(new_rating)
        
    dataset.raw_ratings = new_raw_ratings

    
    return dataset


# %% [markdown]
# ------------------------------------------------------------------

# %% [markdown]
# ### Loading and subsetting datasets

# %%
def subset_dataset(dataset, min_ratings_pu, max_ratings_pu, size): 
    return reindex_dataset(limit_dataset_number(limit_dataset_to_users_w_between_ratings(dataset, min_ratings_pu, max_ratings_pu), size))


# %%
def create_data_subsets(main_dataset, name_prefix, data_subsetting_params, use_multiprocess=True, max_cpus=7):

    start_t = time.time()
    if use_multiprocess:
        with multiprocessing.Pool(processes= min(max_cpus, len(data_subsetting_params) ) ) as pool:        
                datasets_list = pool.starmap(subset_dataset, data_subsetting_params )
    else: 
        datasets_list = list(starmap(subset_dataset, data_subsetting_params ))

    end_t = time.time()
    print("\n%.2f seconds elapsed \n" % (end_t - start_t) )

    subsetted_datasets = {}

    for dataset in datasets_list:
        subsetted_datasets[name_prefix + dataset_desc_str(dataset)] = dataset
        
    return subsetted_datasets



# %%
jester = Dataset.load_builtin('jester')
print("loaded jester")
jester = rescale_dataset(jester)

# parameters are: (dataset, min_ratings_per_user, max_ratings_per_user, total_ratings)
jester_data_subsetting_params =  [ 
                                    (jester, 127, 150, 50000), 
                                    (jester, 116, 126, 50000),     
                                    (jester, 35 , 40 , 50000), 
                                    (jester, 26 , 34 , 20000),  
                                    (jester, 16 , 25 , 20000),     
                                    (jester, 1  , 15 , 20000)     
                                 ] 

jester_datasets = create_data_subsets(jester, "jester", jester_data_subsetting_params)

del jester

# %%
movielens = Dataset.load_builtin('ml-1m')
print("loaded movielens")

movielens_data_subsetting_params = [ 
                                    (movielens, 160, 10000, 25000 ), 
                                    (movielens, 100, 120  , 30000 ),     
                                    (movielens, 38 , 90   , 100000), 
                                    (movielens, 0 , 28    , 15000 )
                                  ] 

movielens_datasets = create_data_subsets(movielens, "movielens", movielens_data_subsetting_params)

del movielens

# %%
datasets = jester_datasets.copy()
datasets.update(movielens_datasets)

print(list(datasets.keys()) )

# %%
print(datasets['movielens_352Rat_25000Tot'].raw_ratings[0:5])


# %% [markdown]
# ----------------------------------------------------------------

# %% [markdown]
# ## Metrics

# %%
def get_ndcg(surprise_predictions, k_highest_scores=None):
    """ 
    Calculates the ndcg (normalized discounted cumulative gain) from surprise predictions, using sklearn.metrics.ndcg_score and scipy.sparse
  
    Parameters: 
    surprise_predictions (List of surprise.prediction_algorithms.predictions.Prediction): list of predictions
    k_highest_scores (positive integer): Only consider the highest k scores in the ranking. If None, use all. 
  
    Returns: 
    float in [0., 1.]: The averaged NDCG scores over all recommendations
  
    """
    
    uids = [int(p.uid) for p in surprise_predictions ]
    iids = [int(p.iid) for p in surprise_predictions ]
    r_uis = [p.r_ui for p in surprise_predictions ]
    ests = [p.est for p in surprise_predictions ]
    
    assert(len(uids) == len(iids) == len(r_uis) == len(ests) )    
    
    sparse_preds = sparse.coo_matrix( (ests, (uids , iids )) )
    sparse_vals = sparse.coo_matrix( (r_uis, (uids , iids )) )
    
    dense_preds = sparse_preds.toarray()
    dense_vals = sparse_vals.toarray()
    
    return ndcg_score(y_true= dense_vals , y_score= dense_preds, k=k_highest_scores) 
    
    

# %%
def rmse_func(predictions): return accuracy.rmse(predictions, verbose=False)
def mae_func(predictions): return accuracy.mae (predictions, verbose=False)
def fcp_func(predictions): return accuracy.fcp (predictions, verbose=False)
def ndcg_func(predictions): return get_ndcg(predictions, k_highest_scores=50)
def ndcg500_func(predictions): return get_ndcg(predictions, k_highest_scores=500)
def ndcgALL_func(predictions): return get_ndcg(predictions, k_highest_scores=None)


# %% [markdown]
# --------------------------------------------------------------------

# %% [markdown]
# ### Plotting 

# %%
def plot_metrics(x_metric, y_metric, res_list, size = 10, use_text = False, highlight_word="baseline"):
    
    plt.style.use('seaborn-whitegrid')

    fig, ax = plt.subplots(figsize=(size, size))

    plt.errorbar(res_list[x_metric], res_list[y_metric] , xerr=res_list[x_metric + '_error'] , yerr=res_list[y_metric + '_error'], fmt='o', color='blue', ecolor='lightblue', elinewidth=5, capsize=0);

    ax.scatter(res_list[x_metric], res_list[y_metric])

    plt.xlabel(x_metric)
    plt.ylabel(y_metric)

    if use_text:
        for i, txt in enumerate( [ s + '_' + t for (s, t) in zip( res_list['algo'], res_list['dataset'])    ] ): 
            if highlight_word.lower() in txt.lower():
                ax.annotate(txt, (res_list[x_metric][i] + 0.0005, res_list[y_metric][i] + 0.0005) )


# %% [markdown]
# ## Main experiment loop

# %%

# %%
# movielens = Dataset.load_builtin('ml-1m')
# print("loaded movielens")

# movielens_data_subsetting_params = [ 
#                                     (movielens, 160, 10000, 2000 ),   
#                                     (movielens, 38 , 90   , 5000)
#                                    ] 

# movielens_datasets = create_data_subsets(movielens, "movielens", movielens_data_subsetting_params)

# del movielens

# %%
# jester = Dataset.load_builtin('jester')
# print("loaded jester")
# jester = rescale_dataset(jester)

# # parameters are: (dataset, min_ratings_per_user, max_ratings_per_user, total_ratings)
# jester_data_subsetting_params =  [ 
#                                     (jester, 127, 150, 5000),  
#                                     (jester, 26 , 34 , 2000)    
#                                  ] 

# jester_datasets = create_data_subsets(jester, "jester", jester_data_subsetting_params)

# del jester

# %%
datasets = jester_datasets.copy()
datasets.update(movielens_datasets)

print(list(datasets.keys()) )

# %%
datasets

# %%

# %%
# from math import sqrt
# from joblib import Parallel, delayed
# Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
# [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

# %%

# %%
#     start_t = time.time()
#     if use_multiprocess:
#         with multiprocessing.Pool(processes= min(7, len(data_subsetting_params) ) ) as pool:        
#                 datasets_list = pool.starmap(subset_dataset, data_subsetting_params )
#     else: 
#         datasets_list = list(starmap(subset_dataset, data_subsetting_params ))

#     end_t = time.time()
#     print("\n%.2f seconds elapsed \n" % (end_t - start_t) )

# %%

# %%
# def run_experiments(algo_dict, dataset_dict, metrics_dict, split_iterator):

#     datasets_split_dict = {}

#     for dataset_name, dataset in dataset_dict.items():

#         splitnum = 1

#         for trainset, testset in list(split_iterator.split(dataset)):

#             dataset_split_name = dataset_name + '|' + str(splitnum)
#             datasets_split_dict[dataset_split_name] =  {'trainset': trainset, 'testset': testset} 

#             splitnum +=1

            
#     experi_names = list(product( list(datasets_split_dict.keys()), list(algo_dict.keys())   )  )
#     experi_params_list =  [ {'dataset': datasets_split_dict[dn], 'dataset_name': dn, 'algorithm': algo_dict[an], 'algorithm_name': an   } for ( dn, an) in experi_names]
    
    
    
    
#     print(experi_names)
    
    

# %%
# metrics_dict = { 'ndcgALL': ndcgALL_func, 'rmse': rmse_func , 'mae': mae_func ,  'fcp': fcp_func  ,  'ndcg': ndcg_func  }
# algo_dict = { 'BaselineOnly': BaselineOnly(verbose=False), 'KNNBasic': KNNBasic(verbose=False)   }
# split_iterator = RepeatedKFold(n_splits=2, n_repeats=1)

# run_experiments(algo_dict, datasets, metrics_dict, split_iterator  )

# %%

# %%

# %%


verbose = True


cv_iterator = RepeatedKFold(n_splits=2, n_repeats=1)

dataset_dict = datasets
algo_dict = { 'BaselineOnly': BaselineOnly(verbose=False)  }
# algo_dict = { 'BaselineOnly': BaselineOnly(verbose=False), 'KNNBasic': KNNBasic(verbose=False)   }

metrics_dict = { 'ndcgALL': ndcgALL_func, 'rmse': rmse_func , 'mae': mae_func ,  'fcp': fcp_func  ,  'ndcg': ndcg_func  }

experi_names = list(product( list(dataset_dict.keys()), list(algo_dict.keys())   )  )
experi_params_list =  [ {'dataset': dataset_dict[dn], 'dataset_name': dn, 'algorithm': algo_dict[an], 'algorithm_name': an   } for ( dn, an) in experi_names]




results = []

total_metric_time = 0
total_algo_time = 0 

for dataset_name in datasets.keys():
    
    for algo_name in algo_dict.keys():
                    

        algo_results_by_metrics = {}       
        metric_results_by_folds_metric = {metric_name:[] for metric_name in metrics_dict.keys() }

        foldnum=0
        
        for trainset, testset in cv_iterator.split(datasets[dataset_name]):
            foldnum += 1
            
            start_t = time.time()
            
            algo = algo_dict[algo_name]
            algo.fit(trainset)
            predictions = algo.test(testset, verbose=False)
            
            end_t = time.time()
            algo_time = end_t - start_t
            total_algo_time += algo_time
            
            if verbose : 
                print("Algo %s fit to dataset %s on fold %s in  %s sec" % (algo_name, dataset_name, foldnum , float('%.2g' % ( algo_time  ))   ) )
            else :
                print('.', end='')
                
        
            for metric_name in metrics_dict.keys():
                
                start_t_metric = time.time()  

                metric_value_fold = metrics_dict[metric_name](predictions)
                metric_results_by_folds_metric[metric_name].append(metric_value_fold)
                
                end_t_metric = time.time()
                
                metric_time = end_t_metric - start_t_metric
                total_metric_time += metric_time
                
                if verbose : print("metric %s calculated in  %.4f sec"   %   ( metric_name , metric_time  )      )
             
    
        for metric_name in metrics_dict.keys():
            
            metric_by_folds = metric_results_by_folds_metric[metric_name]
            
            algo_results_by_metrics[metric_name] = np.mean( metric_by_folds )
            algo_results_by_metrics[metric_name + '_error'] = scipy.stats.norm.ppf(0.5*(0.9+1)) * scipy.stats.sem(  metric_by_folds )
            
  
        algo_results = {'dataset' : dataset_name ,  'algo': algo_name }  
        algo_results.update(algo_results_by_metrics)
        results.append(  algo_results  )

# %% [markdown]
# -------------------------------------------------------------------------------------

# %%
res_list_main = ( lambda LD : {k: [dic[k] for dic in LD] for k in LD[0]} )  (results) 

# %%
print(res_list_main)


# %%

# %%

# %%
def corrplot_of_corrdfr(corrdfr):
    cmap = 'YlGnBu'
    ax = sns.heatmap( corrdfr,  cmap=cmap, square=True )
    ax.set_xticklabels( ax.get_xticklabels(), rotation=45, horizontalalignment='right');


# %%
metric_names = metrics_dict.keys()
dataset_names = datasets.keys()

# %%
df = pd.DataFrame( res_list_main ) 
df['experiment_name'] = df.algo.map(str) + "_" + df.dataset
df = df.set_index('experiment_name')

df = df[metric_names]

corrs_df = df.corr()
corrs_df

# %%
corrplot_of_corrdfr(corrs_df )

# %%
plot_metrics('rmse', 'ndcg', res_list_main)

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
# verbose = True
# n_repeats = 1


# algo_fns = [ BaselineOnly(verbose=False),  KNNBasic(verbose=False)   ]


# # algo_fns = [ BaselineOnly(verbose=False), 
# #             KNNBasic(verbose=False), KNNWithZScore(verbose=False), KNNWithMeans(verbose=False),  
# #             SlopeOne(), CoClustering(verbose=False), NMF(verbose=False),  SVD(verbose=False), SVDpp(verbose=False) ] 


# algo_names = [type(a).__name__ for a in algo_fns]
# algo_fn_list = dict(zip(algo_names, algo_fns))

# metric_fn_list = { 'ndcgALL': ndcgALL_func, 'rmse': rmse_func , 'mae': mae_func ,  'fcp': fcp_func  , 'ndcg500': ndcg500_func, 'ndcg': ndcg_func  }
# metric_fn_list = { 'ndcgALL': ndcgALL_func, 'rmse': rmse_func , 'mae': mae_func ,  'fcp': fcp_func  ,  'ndcg': ndcg_func  }


# results = []

# total_metric_time = 0
# total_algo_time = 0 

# for dataset_name in datasets.keys():
    
#     for algo_name in algo_fn_list.keys():
                    

#         algo_results_by_metrics = {}       
#         metric_results_by_folds_metric = {metric_name:[] for metric_name in metric_fn_list.keys() }

#         foldnum=0
        
#         kf = RepeatedKFold(n_splits=2, n_repeats=n_repeats)
        
#         for trainset, testset in kf.split(datasets[dataset_name]):
#             foldnum += 1
            
#             start_t = time.time()
            
#             algo = algo_fn_list[algo_name]
#             algo.fit(trainset)
#             predictions = algo.test(testset, verbose=False)
            
#             end_t = time.time()
#             algo_time = end_t - start_t
#             total_algo_time += algo_time
            
#             if verbose : 
#                 print("Algo %s fit to dataset %s on fold %s in  %s sec" % (algo_name, dataset_name, foldnum , float('%.2g' % ( algo_time  ))   ) )
#             else :
#                 print('.', end='')
                
        
#             for metric_name in metric_fn_list.keys():
                
#                 start_t_metric = time.time()  

#                 metric_value_fold = metric_fn_list[metric_name](predictions)
#                 metric_results_by_folds_metric[metric_name].append(metric_value_fold)
                
#                 end_t_metric = time.time()
                
#                 metric_time = end_t_metric - start_t_metric
#                 total_metric_time += metric_time
                
#                 if verbose : print("metric %s calculated in  %.4f sec"   %   ( metric_name , metric_time  )      )
             
    
#         for metric_name in metric_fn_list.keys():
            
#             metric_by_folds = metric_results_by_folds_metric[metric_name]
            
#             algo_results_by_metrics[metric_name] = np.mean( metric_by_folds )
#             algo_results_by_metrics[metric_name + '_error'] = scipy.stats.norm.ppf(0.5*(0.9+1)) * scipy.stats.sem(  metric_by_folds )
            
  
#         algo_results = {'dataset' : dataset_name ,  'algo': algo_name }  
#         algo_results.update(algo_results_by_metrics)
#         results.append(  algo_results  )

# %%

# %%

# %%

# %%

# %%

# %%
# print(total_metric_time)
# print(total_algo_time)

# %%
# print(total_metric_time)
# print(total_algo_time)

# %% [markdown]
# time without reindexing is mostly spend on metrics:
#
# print(total_metric_time)
# 247.71
# print(total_algo_time)
# 53.18
#
# time with reindexing is split, but slightly more on algorithms:
#
# print(total_metric_time)
# 36.52
# print(total_algo_time)
# 52.7
#
#
#
#
#

# %%

# %%
