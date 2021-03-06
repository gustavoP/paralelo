#===========================================
# SparkR - 2 Workers
# Antes de rodar esse script iniciar o Spark
# com o shell-script: 
# ./start-spark.sh
#==========================================

# Configuração e bibliotecas
Sys.setenv(SPARK_HOME = "/usr/local/spark")
library(SparkR)
library(tictoc)

# Spark com 2 workers 
sparkR.session(master = "spark://jeff-thinkpad:7077", appName = "SparkR",
               sparkHome = Sys.getenv("SPARK_HOME") )

# carga de arquivo csv  
tic("carregar_dados")
teste <- read.df("/home/jeff/datasets/vra_wu/vra_wu.csv", "csv", header = "true")
toc()
# load_data 

#preprocessar
tic("selecionar_coluna")
airlines<-select(teste, teste$airline)
toc()

#remover
tic("remover_na")
airlines<-dropna(airlines)
toc()

#contar
tic("contar")
results <- summarize(groupBy(airlines,airlines$airline), count(airlines$airline))
#resultst <- summarize(groupBy(dias,dias), count(dias))
toc()

#collect
tic("coletar_resultados")
collect(results)
toc()



