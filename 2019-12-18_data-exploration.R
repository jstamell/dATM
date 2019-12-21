setwd("./Programming_projects/CU_F19/graph/Project/data/")

library(data.table)
library(ggplot2)
library(stringr)

train_meta <- fread('train-track-meta.csv')
test_meta <- fread('test-track-meta.csv')
train_meta_used <- fread('../train-data-used.csv')

### Number songs by year

songs_by_year <- train_meta[,.N,by=year]

ggplot(songs_by_year) + 
  geom_col(aes(x=year,y=N)) +
  scale_x_continuous(limits = c(1950,2012))

songs_by_year[year==0,N]/songs_by_year[,sum(N)]

songs_by_year[year>=1970,sum(N)]
songs_by_year[year>1980,sum(N)]

### Looking at songs with no year
train_meta[year==0]
train_meta[str_detect(title,'One And Only')]

### Number of years that artists are active

artist_active_years <- train_meta[year>=1970,.N,by=.(artist_id,year)][,.(act_years=.N),by=artist_id][,.N,by=act_years]

ggplot(artist_active_years) +
  geom_col(aes(x=act_years,y=N))

train_meta[year>=1970 & artist_id=='ARZ5H0P1187B98A1DD']

train_meta[str_detect(artist_name,'Snoop')]

artist_active_years[act_years<=5,sum(N)]/artist_active_years[,sum(N)]
artist_active_years[act_years>5,sum(N)]

### Number of songs by artist
num_songs_by_artist <- train_meta[year>=1970,.(num_songs=.N),by=.(artist_id)][,.N,by=num_songs]

ggplot(num_songs_by_artist) +
  geom_col(aes(x=num_songs,y=N))

num_songs_by_artist[num_songs<=10,sum(N)]/num_songs_by_artist[,sum(N)]
num_songs_by_artist[num_songs>10,sum(N)]

### Number of artists
train_meta[year>=1970,length(unique(artist_id))]

### Artist active duration
artist_active_duration <- train_meta[year>=1970,.(duration=max(year)-min(year)+1),by=artist_id][,.N,by=duration]
          
ggplot(artist_active_duration) +
  geom_col(aes(x=duration,y=N))

artist_active_duration[duration<=10,sum(N)]/artist_active_duration[,sum(N)]

train_meta[year>=1970,.(duration=max(year)-min(year)+1),by=artist_id][duration==40]

train_meta[artist_id=='AROL34R1187B99510D']


####

test_meta[year >= 1970 & year <=1995,.N] #4639 songs in time period in test data set

test_artists <- test_meta[year >= 1970 & year <=1995,.N,by=artist_id]

train_meta_used[year >= 1970 & year <=1995 & (artist_id %in% test_artists[,artist_id])]

merge(train_meta_used[,.N,by=artist_id], test_artists, by="artist_id")
