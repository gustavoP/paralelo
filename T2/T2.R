library(SparkR)
library(sparklyr)
library(tictoc)

Sys.setenv(SPARK_HOME = "/home/gustavo/spark/spark-2.3.2-bin-hadoop2.7")
Sys.setenv(SPARK_HOME_VERSION='2.3.2')
#sc = spark_connect(master = "local")

sparkR.session()
#sparkR.session(master = "local[*]", sparkConfig = list(spark.driver.memory = "4g"))


vra_wu <- read.df("/home/gustavo/Desktop/vra_wu.csv", "csv", header = "true")#, inferSchema = "true", na.strings = "NA")
aerodromos = read.df("/home/gustavo/Desktop/Glossario_de_Aerodromo.csv","csv", header = "true")


createOrReplaceTempView(vra_wu, "vra_wu")
createOrReplaceTempView(aerodromos, "aerodromos")


test = sql("SELECT 
    origin, 
    COUNT(origin) as CountOrigem 
    FROM vra_wu
    INNER JOIN aerodromos a1 ON vra_wu.origin = a1.Sigla_OACI
    INNER JOIN aerodromos a2 ON vra_wu.destiny = a2.Sigla_OACI
WHERE (year(depart_expect) == 2017 OR year(arrival_expect) ==2017)
AND (departure_delay >15 AND departure_delay <240)
AND status == 'REALIZADO'
AND a1.Pais == 'BRASIL' 
AND a2.Pais =='BRASIL'
GROUP BY vra_wu.origin")




tic("head")
head(test)
toc()

