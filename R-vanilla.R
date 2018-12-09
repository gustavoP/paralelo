#===========================================
# R "vanilla"
#==========================================

# Configuração e bibliotecas
library(tictoc)
library(dplyr)

# carga de arquivo csv
tic("carregar_dados")
teste <- read.csv(file = "/home/jeff/datasets/vra_wu/vra_wu.csv", header = TRUE, sep = ",")
toc()

#preprocessar
tic("selecionar_coluna")
airlines<-teste %>% select(airline)
toc()

#remover
tic("remover_na")
airlines<-na.omit(airlines)
toc()

#contar
tic("contar")
#results <- summarize(groupBy(airlines,airlines$airline), count(airlines$airline))
results <- airlines %>% group_by(airline) %>% count(airline)
toc()

#collect
tic("coletar_resultados")
results
toc()







