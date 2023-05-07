# rm(list=ls(all=TRUE))

setwd("C:/Users/rong/Desktop/NKUST/R/2020碩一期末報告R")
getwd()

library(dplyr)
library(rio)
library(magrittr)
library(ggplot2)
library(plotly)
library(data.table)

happy2020 <- fread("2020WorldHappinessReport.csv")
str(happy2020)
dim(happy2020)
# head(happy2020)
# View(happy2020)


happy2020DT <- data.table(happy2020)
happy2020[grepl("Asia", `Regional indicator`)]
# 把亞洲國家篩出來
asiaHappy2020DT <- happy2020DT[grepl("Asia", `Regional indicator`)] 
head(asiaHappy2020DT$`Country name`)
tail(asiaHappy2020DT$`Country name`)

#亞洲地區幸福指數前五名國家
asiaHappy2020Top5 <- asiaHappy2020DT %>% arrange(desc(`Ladder score`)) %>% slice_head(n = 5)

#Fig
#亞洲地區幸福指數前五名國家
asiaTop5Fig <- ggplotly(ggplot(asiaHappy2020Top5, aes( x = `Country name`, y = `Ladder score`)) +
                          geom_bar(fill='#6A6AFF', col='#6A6AFF', stat = 'identity') +
                          ggtitle("The happiest Asia Country Top5")) ; asiaTop5Fig

#亞洲地區幸福指數前五名國家之Logged GDP per capita圖
asiaTop5GDPFig <- ggplotly(ggplot(asiaHappy2020Top5, aes( x = `Country name`, y = `Logged GDP per capita`)) +
                          geom_bar(fill='#6A6AFF', col='#6A6AFF', stat = 'identity') +
                          ggtitle("The happiest Asia Country Top5-Logged GDP per capita")) ; asiaTop5GDPFig

#亞洲地區幸福指數前五名國家之Social support圖
asiaTop5SocialFig <- ggplotly(ggplot(asiaHappy2020Top5, aes( x = `Country name`, y = `Social support`)) +
                          geom_bar(fill='#6A6AFF', col='#6A6AFF', stat = 'identity') +
                          ggtitle("The happiest Asia Country Top5- Social support")) ; asiaTop5SocialFig

#亞洲地區幸福指數前五名國家之Healthy life expectancy圖
asiaTop5HealthyFig <- ggplotly(ggplot(asiaHappy2020Top5, aes( x = `Country name`, y = `Healthy life expectancy`)) +
                          geom_bar(fill='#6A6AFF', col='#6A6AFF', stat = 'identity') +
                          ggtitle("The happiest Asia Country Top5-Healthy life expectancy")) ; asiaTop5HealthyFig

#亞洲地區幸福指數前五名國家之Freedom to make life choices圖
asiaTop5FreedomFig <- ggplotly(ggplot(asiaHappy2020Top5, aes( x = `Country name`, y = `Freedom to make life choices`)) +
                                    geom_bar(fill='#6A6AFF', col='#6A6AFF', stat = 'identity') +
                                    ggtitle("The happiest Asia Country Top5 - Freedom ")) ; asiaTop5FreedomFig

#亞洲地區幸福指數前五名國家之Generosity圖
asiaTop5GenerosityFig <- ggplotly(ggplot(asiaHappy2020Top5, aes( x = `Country name`, y = `Generosity`)) +
                                    geom_bar(fill='#6A6AFF', col='#6A6AFF', stat = 'identity') +
                                    ggtitle("The happiest Asia Country Top5 - Generosity")) ; asiaTop5GenerosityFig

#亞洲地區幸福指數前五名國家之Perceptions of corruption圖
asiaTop5CorruptionFig <- ggplotly(ggplot(asiaHappy2020Top5, aes( x = `Country name`, y = `Perceptions of corruption`)) +
                                    geom_bar(fill='#6A6AFF', col='#6A6AFF', stat = 'identity') +
                                    ggtitle("The happiest Asia Country Top5 - Corruption")) ; asiaTop5CorruptionFig