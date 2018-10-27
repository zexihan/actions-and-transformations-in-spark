package ats

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SQLContext
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._
import org.apache.log4j.LogManager
import org.apache.log4j.Level

case class Edge(v1: Int, v2: Int)
case class PR(v: Int, r: Double)

object PageRank_Dataset {
  
  def main(args: Array[String]) {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger
    
    val conf = new SparkConf().setAppName("PageRank RDD")
    conf.set("spark.eventLog.enabled", "true")
    conf.set("spark.eventLog.dir", "logs")
    val sc = new SparkContext(conf)
    
    val spark: SparkSession = SparkSession.builder
    .config(conf)
    .getOrCreate;

    import spark.implicits._

		// delete output directory, only to ease local development; will not work on AWS. ===========
    val hadoopConf = new org.apache.hadoop.conf.Configuration
    val hdfs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
    try { hdfs.delete(new org.apache.hadoop.fs.Path(args(1)), true) } catch { case _: Throwable => {} }
		// ================
    
    // initialize graph
    val k = 100
    var g = Seq[Edge]()
    for (i <- 1 to k*k) {
      if (i % k != 0) {
        g = g :+ Edge(i, i+1)
      } else{
        g = g :+ Edge(i, 0)
      }
    }
    val graph = g.toDS.coalesce(4)
    	
    // initialize ranks
    var r = Seq[PR]()
    for (i <- 1 to k*k) {
      r = r :+ PR(i, 1.0/(k*k))
    }
    r = r :+ PR(0, 0.0)
    var ranks = r.toDS.coalesce(4)
    
    // run pagerank iteratively
    var pr_sum = Array[Any]()
    for (i <- 1 to 10) {
      val Temp = graph.joinWith(ranks, graph("v1") === ranks("v")).flatMap(joined => if (joined._1.v1 % k == 1) List(PR(joined._1.v1, 0.0), PR(joined._1.v2, joined._2.r)) else List(PR(joined._1.v2, joined._2.r)))
      val Temp2DF = Temp.groupBy("v").agg(sum(Temp("r")))
      val Temp2: Dataset[PR] = Temp2DF.map{row => PR(row.getAs[Int](0), row.getAs[Double](1))}
      val deltaTemp = Temp2.filter(tuple => tuple.v == 0).map(tuple => tuple.r).collect
      val delta = deltaTemp(0)
      ranks = Temp2.map(tuple => if (tuple.v != 0) PR(tuple.v, tuple.r + delta/(k*k)) else PR(tuple.v, 0.0))
      pr_sum = pr_sum :+ (ranks.agg(sum($"r")).collect())(0)(0)
    }
    
    // debug
    print(ranks.explain)
    
    // report the pagerank sum for each of the 10 iterations
    logger.setLevel(Level.WARN)
    logger.warn(pr_sum.mkString("\n"))
    
    // report the final pageranks of the first 101 vertices
    sc.parallelize(ranks.orderBy("v").take(101).toSeq).saveAsTextFile(args(1))
    
  }
}

