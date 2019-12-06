################################
# CAPSTONE - MovieLens - Brandon Rufino
################################


################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(purrr)) install.packages("purrr", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages("dslabs", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos ="http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(tidyr)) install.packages("tidyr", repos = "http://cran.us.r-project.org")
if(!require(recommenderlab)) install.packages("recommenderlab", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# CREATE DATA
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#LOAD DATASET
# edx <- miceadds::load.Rdata2( filename="data/edx.Rdata")
# validation <- miceadds::load.Rdata2( filename="data/validation.Rdata")


################################
# MODEL EXPLORATION
################################

# PART ONE: CREATE TRAINING AND TESTING SET FOR SELECTING BEST MODEL
set.seed(755, sample.kind = "Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1,
                                  p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]


# PART TWO: DEFINE MODELS

#MODEL 1: MOVIE RATING EFFECT

# calculate the average of all ratings of the edx set
mu <- mean(train_set$rating)

#calculate b_i on the training set
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# predicted ratings
predicted_ratings_bi <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

# set any movies who did not have a rating in training_set to zero
predicted_ratings_bi[is.na(predicted_ratings_bi)]<-mu

#MODEL 2: USER/MOVIE RATING EFFECT

# calculate b_u using the training set 
user_avgs <- train_set %>%  
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#predicted ratings
predicted_ratings_bu <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

predicted_ratings_bu[is.na(predicted_ratings_bu)]<-mu

#MODEL 3: USER/MOVIE/GENRE RATING EFFECT

# calculate b_g using the training set 
genre_avgs <- train_set %>%  
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

#predicted ratings
predicted_ratings_bg <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  .$pred

predicted_ratings_bg[is.na(predicted_ratings_bg)]<-mu

#MODEL 4: REGULARIZATION

lambdas <- seq(0, 10, 0.25)

regularize<- function(l){
  
  mu_reg <- mean(train_set$rating)
  
  b_i_reg <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i_reg = sum(rating - mu_reg)/(n()+l))
  
  b_u_reg <- train_set %>% 
    left_join(b_i_reg, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u_reg = sum(rating - b_i_reg - mu_reg)/(n()+l))
  
  b_g_reg <- train_set %>% 
    left_join(b_i_reg, by="movieId") %>%
    left_join(b_u_reg, by='userId') %>%
    group_by(genres) %>%
    summarize(b_g_reg = sum(rating - b_i_reg - b_u_reg - mu_reg)/(n()+l))
  
  predicted_ratings_b_i_u_g <- 
    test_set %>% 
    left_join(b_i_reg, by = "movieId") %>%
    left_join(b_u_reg, by = "userId") %>%
    left_join(b_g_reg, by= 'genres') %>%
    mutate(pred = mu_reg + b_i_reg + b_u_reg + b_g_reg) %>%
    .$pred
  
  predicted_ratings_b_i_u_g[is.na(predicted_ratings_b_i_u_g)]<-mu
  
  return(RMSE(predicted_ratings_b_i_u_g, test_set$rating))
}

# PART THREE: DEFINE RMSE

RMSE <- function(predicted_ratings, true_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# PART FOUR: EVALUATE, AND SELECT BEST

rmse_model1 <- RMSE(predicted_ratings_bi,test_set$rating)  
rmse_model1 
rmse_model2 <- RMSE(predicted_ratings_bu,test_set$rating)
rmse_model2
rmse_model3 <- RMSE(predicted_ratings_bg,test_set$rating)
rmse_model3
rmses <- sapply(lambdas, regularize)

lambda <- lambdas[which.min(rmses)]

rmse_model4 <- min(rmses)
rmse_model4

################################
# FINAL EVALUATION
################################
#train with edx
#final evaluation with validation
mu_reg <- mean(edx$rating)
b_i_reg <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i_reg = sum(rating - mu_reg)/(n()+lambda))

b_u_reg <- edx %>% 
  left_join(b_i_reg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u_reg = sum(rating - b_i_reg - mu_reg)/(n()+lambda))

b_g_reg <- edx %>% 
  left_join(b_i_reg, by="movieId") %>%
  left_join(b_u_reg, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g_reg = sum(rating - b_i_reg - b_u_reg - mu_reg)/(n()+lambda))

predicted_ratings_b_i_u_g <- 
  validation %>% 
  left_join(b_i_reg, by = "movieId") %>%
  left_join(b_u_reg, by = "userId") %>%
  left_join(b_g_reg,by='genres') %>%
  mutate(pred = mu_reg + b_i_reg + b_u_reg+ b_g_reg) %>%
  .$pred


predicted_ratings_b_i_u_g[is.na(predicted_ratings_b_i_u_g)]<-mu
print("Final Evaluation. RMSE for validation data-set:")
RMSE(predicted_ratings_b_i_u_g, validation$rating)