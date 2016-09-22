package org.apache.mahout.sparkbindings.mlsystems

import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.Vector
import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import drm._
import RLikeDrmOps._
import org.apache.mahout.sparkbindings._
import org.apache.spark.{SparkConf, SparkContext}

object Example extends App {

  val sparkConf = new SparkConf()
  sparkConf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
  val sparkContext = new SparkContext("local[2]", "Example", sparkConf)

  try {

    implicit val sc = new SparkDistributedContext(sparkContext)

    val data = drmParallelize(dense(
      (2, 2, 10.5, 10, 29.509541), // Apple Cinnamon Cheerios
      (1, 2, 12, 12, 18.042851), // Cap'n'Crunch
      (1, 1, 12, 13, 22.736446), // Cocoa Puffs
      (2, 1, 11, 13, 32.207582), // Froot Loops
      (1, 2, 12, 11, 21.871292), // Honey Graham Ohs
      (2, 1, 16, 8, 36.187559), // Wheaties Honey Gold
      (6, 2, 17, 1, 50.764999), // Cheerios
      (3, 2, 13, 7, 40.400208), // Clusters
      (3, 3, 13, 4, 45.811716)), // Great Grains Pecan
      numPartitions = 2)

    val betaHat = dridge(data, 0.01)

    println(betaHat)

    sc.engine.logicalRewrites.foreach { rewrite =>

      println(rewrite)
    }

  } finally {
    sparkContext.close()
  }



  def dridge(data: DrmLike[Int], lambda: Double): Vector = {

    val drmX = data(::, 0 until data.ncol - 1) cbind 1
    val y = data.collect(::, data.ncol - 1)

    val drmXtX = drmX.t %*% drmX
    val drmXty = drmX.t %*% y

    val XtX = drmXtX.collect
    val Xty = drmXty.collect(::, 0)

    XtX.diagv += lambda

    solve(XtX, Xty)
  }

}
