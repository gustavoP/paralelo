#===========================================
# Sparklyr - 2 Workers
# Antes de rodar esse script iniciar o Spark
# com o shell-script: 
# ./start-spark.sh
#==========================================

# Configuração e bibliotecas
Sys.setenv(SPARK_HOME = "/usr/local/spark")
library(tictoc)
library(sparklyr)
library(dplyr)

# Spark com 2 workers 
cfg <- spark_config()
sc <- spark_connect(master ="spark://jeff-thinkpad:7077", config = cfg)

# carga de arquivo csv
tic("carregar_dados")
teste <- spark_read_csv(sc, name = "wru", path = "/home/jeff/datasets/vra_wu/vra_wu.csv", header = TRUE, delimiter = ",")
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
results <- airlines %>% group_by(airline) %>% summarise(count())
toc()

#collect
tic("coletar_resultados")
collect(results)
toc()

