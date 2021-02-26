# Spotify-Song-Popularity-Prediction

This project aims to understand the features that *hit* songs have in common. In particular, I was interested in knowing if a song's artist, the genre of the song and, its other musical features like danceability, energy, loudness among others can help distinguish *hit* and *non-hit* songs. To achieve this, I trained a logitistic regression model and compared the results of this model with the non-parametric classification method, random forests. Since the logistic regression model is more interpretable, I have drawn inferences from it. From the model, it can inferred that the popularity of a song's artist and the song genre does highly influence the odds of a song getting *hit*. Another influential music feature is danceability. Songs that are more danceable have higher odds of becoming *hit*.

# Data
In order to do an analysis, I needed music data with music related features, artist popularity, song_genre and, other metadata of the songs. Spotify's music data that it provides via its developer APIs seemed like the right place to start to create this dataset.
More information can be found [here](https://developer.spotify.com/documentation/web-api/) 

# Visulization app
I built a RShiny visuzlization dashboard for this project. The dashboard can be viewed [here](https://ashwinimarathe.shinyapps.io/musicvisualization/)
