library(dplyr)
library(ggplot2)
library(arm)
library(pROC)
library(e1071)
library(caret)
library(randomForest)
library(ramify)

## read file
df <- read.csv("all_genre_5000.csv")
df$hit <- ifelse(df$popularity>50,1,0)

## remove non unique (song_name, artist) pair
df_multiple <- list(df %>% group_by(song_name, artist_name) %>% tally() %>% filter(n<2))
x <- df[df$song_name %in% df_multiple[[1]]$song_name,]
songs <- x %>% arrange(song_name, artist_name)
songs$hit <- ifelse(songs$popularity>50, 1, 0)

## remove romance and progressive metal genre
songs <- songs %>% filter(song_genre != 'romance' & song_genre!='progressive metal')

## EDA

ggplot(songs, aes(x=popularity)) + geom_histogram()

ggplot(songs, aes(y=popularity, x=artist_popularity)) + geom_point()

ggplot(songs, aes(y=popularity, x= (song_genre))) + geom_boxplot()

ggplot(songs, aes(y=popularity, x= as.factor(mode))) + geom_boxplot()

ggplot(songs, aes(y=popularity, x= as.factor(key))) + geom_boxplot()

ggplot(songs, aes(y=popularity, x= (duration_ms))) + geom_point()

ggplot(songs, aes(y=popularity, x= as.factor(time_signature))) + geom_boxplot()

ggplot(songs, aes(y=popularity, x= valence)) + geom_point()

binnedplot(y=songs$hit,songs$duration_ms,xlab="duration_ms",ylim=c(0,0.3),col.pts="navy",
           ylab ="hit?",
           col.int="white")

binnedplot(y=songs$hit,songs$instrumentalness,xlab="instrumentalness",ylim=c(0,0.3),col.pts="navy",
           ylab ="hit?",
           col.int="white")

binnedplot(y=songs$hit,songs$artist_popularity,xlab="artist_popularity",ylim=c(0,0.3),col.pts="navy",
           ylab ="hit?",
           col.int="white")

binnedplot(y=songs$hit,songs$loudness,xlab="loudness",ylim=c(0,0.3),col.pts="navy",
           ylab ="hit?",
           col.int="white")

binnedplot(y=songs$hit,songs$acousticness,xlab="acousticness",ylim=c(0,0.3),col.pts="navy",
           ylab ="hit?",
           col.int="white")

binnedplot(y=songs$hit,songs$danceability,xlab="danceability",ylim=c(0,0.3),col.pts="navy",
           ylab ="hit?",
           col.int="white")

binnedplot(y=songs$hit,songs$energy,xlab="energy",ylim=c(0,0.3),col.pts="navy",
           ylab ="hit?",
           col.int="white")

binnedplot(y=songs$hit,songs$liveness,xlab="liveness",ylim=c(0,0.3),col.pts="navy",
           ylab ="hit?",
           col.int="white")

binnedplot(y=songs$hit,songs$valence,xlab="valence",ylim=c(0,0.3),col.pts="navy",
           ylab ="hit?",
           col.int="white")

features <- songs

# feature engineering based on EDA

features$liveness <- ifelse(features$liveness<0.5,0,1)
features$instrumentalness <- as.factor(ifelse(features$instrumentalness<0.2, 0,1))
features$energy <- (features$energy-mean(features$energy))/sd(features$energy)
features$acousticness <- (features$acousticness - mean(features$acousticness))/sd(features$acousticness)
features$danceability <- (features$danceability - mean(features$danceability))/sd(features$danceability)

features$loudness <- (features$loudness - mean(features$loudness))/sd(features$loudness)

features$valence_sq <- (features$valence)^2
features$valence <- (features$valence - mean(features$valence))/sd(features$valence)
features$speechiness <- (features$speechiness - mean(features$speechiness))/sd(features$speechiness)
features$tempo <- (features$tempo - mean(features$tempo))/sd(features$tempo)
features$time_signature <- as.factor(features$time_signature)

features$duration_ms_1 <- cut(songs$duration_ms, breaks = c(0,200000, 400000, Inf), labels = c(1,2,3))
features$artist_popularity_1 <- ifelse(songs$artist_popularity<38,0,songs$artist_popularity)

features$artist_popularity <- (features$artist_popularity - mean(features$artist_popularity))/sd(features$artist_popularity)
features$artist_popularity_1 <- (features$artist_popularity_1 - mean(features$artist_popularity_1))/sd(features$artist_popularity_1)
features$artist_pop_dance <- features$artist_popularity_1 * features$danceability
features$release_date <- as.character(features$release_date)
features$day <- weekdays(as.Date(features$release_date))
features$month <- month(as.Date(features$release_date))

# handling missing values in day and month
features[is.na(features$day),]$day <- median(features$day, na.rm = TRUE)
features[is.na(features$month),]$month <- median(features$month, na.rm = TRUE)
# model building

# test train split

smp_size <- floor(0.8 * nrow(features))

## set the seed to make your partition reproducible
set.seed(12) #12
train_ind <- sample(seq_len(nrow(features)), size = smp_size)

train <- features[train_ind, ]
test <- features[-train_ind, ]

# model1
model1 <- glm(hit ~ acousticness+danceability+energy+liveness+loudness+speechiness+key+mode+tempo
              +song_genre+artist_popularity_1+duration_ms_1+day+month+valence_sq
              , data=train, family = binomial)
summary_model1 <- summary(model1)

roc(train$hit,fitted(model1),plot=T,print.thres="best",legacy.axes=T,
    print.auc =T,col="red3")
Conf_mat1 <- confusionMatrix(as.factor(ifelse(fitted(model1) >= 0.3, "1","0")),
                             as.factor(train$hit),positive = "1")

# model2

model4 <- glm(as.factor(hit) ~ acousticness+danceability+energy+liveness+loudness+
                speechiness+mode+tempo+song_genre:loudness+day+valence
              +(song_genre)+artist_popularity_1+duration_ms_1, family=binomial(link="logit"),data=train)
summary(model4)

summary_model4 <- summary(model4)
rawresiduals <- residuals(model4, "resp")

# calculating threshold for highest f1 score
f1 = c()
thres = seq(0.05,0.95, by=0.05)
for (i in thres){
  f1 = append(f1, F1_Score(as.factor(ifelse(predict(model4, newdata=train, type="response") >= i, "1","0")),
                           as.factor(train$hit),positive = "1"))
}
optimal_thres <- thres[which.max(f1)]

Conf_mat2 <- confusionMatrix(as.factor(ifelse(predict(model4, newdata=train, type="response") >= optimal_thres, "1","0")),
                             as.factor(train$hit),positive = "1")

Conf_mat2_test <- confusionMatrix(as.factor(ifelse(predict(model4, newdata=test, type="response") >= optimal_thres, "1","0")),
                                  as.factor(test$hit),positive = "1")

# residual plots
par(mfrow=c(1,2))
binnedplot(x=fitted(model4),y=rawresiduals,xlab="Pred. probabilities",
           col.int="red4",ylab="Avg. residuals",main="Figure3a: Binned residual plot",col.pts="navy")
binnedplot(x[train_ind,]$artist_popularity,y=rawresiduals,xlab="Artist Popularity",
           col.int="red4",ylab="Avg. residuals",main="Figure3b: Binned residual plot",col.pts="navy")






