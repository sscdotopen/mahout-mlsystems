package org.apache.mahout.sparkbindings.mlsystems

import java.util.UUID

import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import drm._
import RLikeDrmOps._
import org.apache.mahout.sparkbindings._
import org.apache.spark.{SparkConf, SparkContext}


object Experiments extends App {

  val numCores = Runtime.getRuntime.availableProcessors
  val numRepetitions = 10

  var outcomes = Seq[String]()

  //val numRows = 1000000
  val numColumns = 20

  for (numRows <- Array(50000, 100000, 250000, 500000, 750000)) {
    //outcomes ++= runLocalExperiment(numRepetitions, numCores, numCores, numRows, numColumns, false, false, false)
    //outcomes ++= runLocalExperiment(numRepetitions, numCores, numCores, numRows, numColumns, true, true, false)
    outcomes ++= runLocalExperiment(numRepetitions, numCores, numCores, numRows, numColumns, true, true, true)
  }

  println("### OUTCOMES ###")
  outcomes.foreach { println }

  def runLocalExperiment(numRepetitions: Int, numWorkers: Int, numPartitions: Int, numRows: Int, numColumns: Int,
    allowAdvancedRewrites: Boolean, allowSmartPhysicalChoices: Boolean, manualCaching: Boolean): Seq[String] = {

    val expId = UUID.randomUUID().toString

    inLocalSpark(numWorkers) { sparkContext =>

      implicit val sc = new SparkDistributedContext(sparkContext)

      val randomInput = Matrices.uniformView(numRows, numColumns, 0xdead)

      assert(randomInput.getFlavor.isDense)
      val generatedData = drmParallelize(randomInput, numPartitions = numPartitions)

      generatedData.dfsWrite(s"/home/ssc/tmp/${expId}.ser")

    }

    (0 until numRepetitions) map { r =>
      var result = ""
      inLocalSpark(numWorkers) { sparkContext =>

        implicit val sc = new SparkDistributedContext(sparkContext)
        sc.engine.allowAdvancedRewrites = allowAdvancedRewrites
        sc.engine.allowSmartPhysicalChoices = allowSmartPhysicalChoices

        val data = drmDfsRead(s"/home/ssc/tmp/${expId}.ser", nrow = numRows, ncol = numColumns).asInstanceOf[DrmLike[Int]]


        val (runtimeA, runtimeB) = dridge(data, 0.01, manualCaching)


        val plans = sc.engine.logicalRewrites.map { _.rewrittenPlan }.mkString("---")
        sc.engine.logicalRewrites.clear()

        result = s"${numWorkers}-${numPartitions}-${numRows}-${numColumns}-${sc.engine.allowAdvancedRewrites}-${manualCaching}: ${runtimeA},${runtimeB} | ${plans}"
      }

      result
    }
  }

  def inLocalSpark(numWorkers: Int)(udf: SparkContext => Unit): Unit = {
    val sparkConf = new SparkConf()
    sparkConf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    sparkConf.set("spark.driver.memory", "6g")
    val sparkContext = new SparkContext(s"local[${numCores}]", "Example", sparkConf)

    try {
      udf(sparkContext)
    } finally {
      sparkContext.close()
    }
  }

  def dridge(data: DrmLike[Int], lambda: Double, manualCaching: Boolean = false): (Long, Long) = {

    val startA = System.currentTimeMillis()
    val drmX = data(::, 0 until data.ncol - 1) cbind 1
    val drmY = data(::, data.ncol - 2 until data.ncol - 1)

    if (manualCaching) {
      drmX.checkpoint(CacheHint.MEMORY_ONLY)
    }

    val drmXtX = drmX.t %*% drmX
    val drmXty = drmX.t %*% drmY

    val XtX = drmXtX.collect
    val endA = System.currentTimeMillis()
    val runtimeA = endA - startA
    val Xty = drmXty.collect
    val runtimeB = System.currentTimeMillis() - endA

    XtX.diagv += lambda

    solve(XtX, Xty)
    (runtimeA, runtimeB)
  }
}
