package ats

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SQLContext
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.LogManager
import org.apache.log4j.Level

object PageRank_RDD {
  
  def main(args: Array[String]) {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger
    
    val conf = new SparkConf().setAppName("PageRank RDD")
    conf.set("spark.eventLog.enabled", "true")
    conf.set("spark.eventLog.dir", "logs")
    val sc = new SparkContext(conf)

		// delete output directory, only to ease local development; will not work on AWS. ===========
    val hadoopConf = new org.apache.hadoop.conf.Configuration
    val hdfs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)
    try { hdfs.delete(new org.apache.hadoop.fs.Path(args(1)), true) } catch { case _: Throwable => {} }
		// ================
    
    // initialize graph
    val k = 100
    var g = Seq[(Int, Int)]()
    for (i <- 1 to k*k) {
      if (i % k != 0) {
        g = g :+ (i, i+1)
      } else{
        g = g :+ (i, 0)
      }
    }
    val graph = sc.parallelize(g, 4)

    // initialize ranks
    var r = Seq[(Int, Double)]()
    for (i <- 1 to k*k) {
      r = r :+ (i, 1.0/(k*k))
    }
    r = r :+ (0, 0.0)
    var ranks = sc.parallelize(r, 4)
    
    // run pagerank iteratively
    var pr_sum = Array[Any]()
    for (i <- 1 to 10) {
      val Temp = graph.join(ranks).flatMap(tuple => if (tuple._1 % k == 1) List((tuple._1, 0.0), tuple._2) else List(tuple._2)) 
      val Temp2 = Temp.groupByKey().map(tuple => (tuple._1, tuple._2.sum))
      val deltaTemp = Temp2.filter(tuple => tuple._1 == 0).map(tuple => tuple._2).collect
      val delta = deltaTemp(0)
      ranks = Temp2.map(tuple => if (tuple._1 != 0) (tuple._1, tuple._2 + delta/(k*k)) else (tuple._1, 0.0))
      pr_sum = pr_sum :+ ranks.map(tuple => tuple._2).sum()
    }
    
    // debug
    print(ranks.toDebugString)
    
    // report the pagerank sum for each of the 10 iterations
    logger.setLevel(Level.WARN)
    logger.warn(pr_sum.mkString("\n"))
    
    // report the final pageranks of the first 101 vertices
    sc.parallelize(ranks.sortByKey().take(101).toSeq).saveAsTextFile(args(1))
    
  }
}