# MovieRecommendationSystem

## Overview

Recommender systems are some of the most important machine learning applications to date. Specifically, recommender systems are relied on heavily by companies to cater great consumer experiences. This can be seen through streaming services such as Netflix as well as retail companies such as Nike. Thus, in this project I will create a movie recommender system that will predict movie ratings for a user.

## Problem statement

For this project we are creating a movie recommendation system using MovieLens dataset. This dataset is a small subset of a much larger dataset with millions of ratings. We are given code to create two datasets: (1) edx and (2) validation. The edx dataset is the dataset in which we will train and test our models. The validation dataset is our 'unseen' data and will be used to report our final RMSE value. The goal of this project is to achieve an RMSE value less than or equal to 0.8649. 

This project will go through several machine learning stages. First, we will state the problem presented by the movielens dataset. We will extrapolate the data from the movielens database into a friendly format which contains information such as users, movies, genres, and their ratings. Second, we will explore the data-set and look at interesting trends. Once data exploration is finished, we will preprocess our data to extrapolate features that we may require for model creation. With our features in hand, we will create several models and predict movie ratings. Lastly, we will choose the best model to run against unseen data and report the final RMSE value.


_I hope you guys enjoy reading this as much as I did creating it!_