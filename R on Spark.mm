<map version="1.0.1">
<!-- To view this file, download free mind mapping software FreeMind from http://freemind.sourceforge.net -->
<node CREATED="1543546425310" ID="ID_708506160" MODIFIED="1543546895315" TEXT="R on Spark">
<node CREATED="1543546783211" ID="ID_321731816" MODIFIED="1543546787148" POSITION="right" TEXT="introdu&#xe7;&#xe3;o">
<node CREATED="1543546871412" ID="ID_1751720760" MODIFIED="1543546874708" TEXT="Resumo"/>
<node CREATED="1543546752936" ID="ID_1391545939" MODIFIED="1543546763597" TEXT="Motiva&#xe7;&#xe3;o"/>
</node>
<node CREATED="1543546835027" ID="ID_967421121" MODIFIED="1543546842779" POSITION="right" TEXT="Referencial Te&#xf3;rico"/>
<node CREATED="1543546798657" ID="ID_1201559135" MODIFIED="1543546805271" POSITION="right" TEXT="Materiais e m&#xe9;todos">
<node CREATED="1543546499745" ID="ID_1718586516" MODIFIED="1543546510721" TEXT="Requisitos e instala&#xe7;&#xe3;o">
<node CREATED="1543547409906" ID="ID_805761688" MODIFIED="1543548481552" TEXT="Spark Local">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="rgb(51, 51, 51)" face="Helvetica Neue, Helvetica, Arial, sans-serif">When connecting to Spark in local mode, Spark starts as a single application simulating a cluster with a single node, this is not a proper computing cluster but is ideal to perform work offline and troubleshoot issues</font>
    </p>
    <p>
      http://therinspark.com/connections.html#connection-local
    </p>
  </body>
</html>
</richcontent>
</node>
<node CREATED="1543548482007" ID="ID_426873928" MODIFIED="1543548563073" TEXT="Spark Standalone">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="rgb(51, 51, 51)" face="Helvetica Neue, Helvetica, Arial, sans-serif" size="16px">Connecting to a Spark Standalone cluster requires the location of the cluster manager&#8217;s master instance, this location can be found in the cluster manager web interface as described in the&#160;</font>
    </p>
  </body>
</html>
</richcontent>
</node>
</node>
<node CREATED="1543547061415" ID="ID_320240677" MODIFIED="1543548352713" TEXT="Configura&#xe7;&#xe3;o no RStudio">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      Como fazer a conex&#227;o com o Spark de dentro do RStudio, os seguintes comandos s&#227;o implementados pelo Sparklyr:
    </p>
    <p>
      library(sparklyr)
    </p>
    <p>
      library(dplyr)
    </p>
    <p>
      sc &lt;- spark_connect(master = &quot;local&quot;)
    </p>
    <p>
      
    </p>
    <p>
      &#201; poss&#237;vel especificar a quantidade de cores (processadores) que deseja utilizar. O padr&#227;o &#233; utilizar todos os dispon&#237;veis. Fa&#231;a isso na linha do comado spark_connect. Ex: para utilizar 2 cores:<br />sc &lt;- spark_connect(master = &quot;local&quot;[2])
    </p>
  </body>
</html>
</richcontent>
</node>
<node CREATED="1543546513541" ID="ID_1372177912" MODIFIED="1543546522571" TEXT="Compara&#xe7;&#xe3;o de desempenho">
<node CREATED="1543546528428" ID="ID_1000814455" MODIFIED="1543548442974" TEXT="processador">
<richcontent TYPE="NOTE"><html>
  <head>
    
  </head>
  <body>
    <p>
      Utilizando apenas 2 cores, conseguimos carregar o arquivo vra-wu em 46.682 segundos.
    </p>
  </body>
</html>
</richcontent>
</node>
<node CREATED="1543546546199" ID="ID_691871292" MODIFIED="1543546548404" TEXT="mem&#xf3;ria"/>
</node>
</node>
<node CREATED="1543546815465" ID="ID_909784442" MODIFIED="1543546829223" POSITION="right" TEXT="Resultados e conclus&#xf5;es"/>
</node>
</map>
