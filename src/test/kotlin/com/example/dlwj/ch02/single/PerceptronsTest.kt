package com.example.dlwj.ch02.single

import com.example.dlwj.ch02.util.GaussianDistribution
import com.example.dlwj.ch02.util.PlotUtil
import javafx.application.Platform
import jetbrains.datalore.base.registration.Disposable
import jetbrains.datalore.plot.MonolithicCommon
import jetbrains.datalore.vis.swing.jfx.DefaultPlotPanelJfx
import jetbrains.letsPlot.geom.geomHistogram
import jetbrains.letsPlot.geom.geomPoint
import jetbrains.letsPlot.intern.Plot
import jetbrains.letsPlot.intern.toSpec
import jetbrains.letsPlot.letsPlot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.toDoubleArray
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import org.junit.jupiter.api.Test
import java.awt.Dimension
import java.awt.GridLayout
import java.util.*
import java.util.concurrent.CountDownLatch
import javax.swing.*
import kotlin.random.Random

// 单层感知器
class PerceptronsTest {
    @Test
    fun main() {
        // 训练数据的数量
        val train_N = 1000
        // 测试数据的数量
        val test_N = 200
        // 输入数据的维度
        val nIn = 2

        // 用于训练的输入数据
        val train_X: NDArray<Double, D2> = mk.zeros<Double>(train_N, nIn)
        // 用于训练的输出数据
        val train_T = IntArray(train_N)

        // 用于测试的输入数据
        val test_X = mk.zeros<Double>(test_N, nIn)
        // 用于测试的数据的实际标记
        val test_T = IntArray(test_N)
        // 模型预测的输出数据
        val predicted_T = IntArray(test_N)
        // 最大迭代次数
        val epochs = 2000
        // 感知器中学习率可以为1
        val learningRate = 1.0
        val rng = Random(1234)
        // 均值-2.0，方差1.0
        val g1 = GaussianDistribution(-2.0, 1.0, rng)
        // 均值2.0，方差1.0
        val g2 = GaussianDistribution(2.0, 1.0, rng)
        // 第一类 [g1.random(), g2.random()]
        // 生成第一类训练数据
        (0 until (train_N / 2 - 1)).forEach {
            train_X[it, 0] = g1.random()
            train_X[it, 1] = g2.random()
            train_T[it] = 1
        }
        // 生成第一类测试数据
        (0 until (test_N / 2 - 1)).forEach {
            test_X[it, 0] = g1.random()
            test_X[it, 1] = g2.random()
            test_T[it] = 1
        }
        // 第二类 [g2.random(), g1.random()]
        ((train_N / 2) until train_N).forEach {
            train_X[it, 0] = g2.random()
            train_X[it, 1] = g1.random()
            train_T[it] = -1
        }
        ((test_N / 2) until test_N).forEach {
            test_X[it, 0] = g2.random()
            test_X[it, 1] = g1.random()
            test_T[it] = -1
        }
        plot(train_X)


        // 分类器
        val classifier = Perceptrons(nIn)
        // 训练模型
        (0..epochs).find {epoch->
            val classified_: Int = (0 until train_N).sumOf { classifier.train(train_X[it], train_T[it], learningRate) }
            println("训练次数：$epoch, 成功分类：${classified_*1.0/train_N}")
            classified_ == train_N
        }
        // 测试
        (0 until test_N).forEach { predicted_T[it] = classifier.predict(test_X[it]) }

        // 评估模型
        val confusionMatrix = mk.zeros<Int>(2, 2)

        // 准确率  所有数据 正确分类比率
        var accuracy = 0.0
        // 精确度  正例 正确分类比率
        var precision = 0.0
        // 召回率  预测为正例 正确分类比率
        var recall = 0.0
        (0 until test_N).forEach {
            if (predicted_T[it] > 0) {
                if (test_T[it] > 0) {
                    accuracy += 1
                    precision += 1
                    recall += 1
                    confusionMatrix[0, 0] += 1
                } else {
                    confusionMatrix[1, 0] += 1
                }
            } else {
                if (test_T[it] > 0) {
                    confusionMatrix[0, 1] += 1
                } else {
                    accuracy += 1
                    confusionMatrix[1, 1] += 1
                }
            }
        }
        accuracy /= test_N
        precision /= confusionMatrix[0, 0] + confusionMatrix[1, 0]
        recall /= confusionMatrix[0, 0] + confusionMatrix[0, 1]
        println(confusionMatrix)
        println(
            """
--------------------------------------
感知器模型评价
--------------------------------------
Accuracy:  ${"%.1f %%".format(accuracy * 100)}
Precision: ${"%.1f %%".format(precision * 100)}
Recall:    ${"%.1f %%".format(recall * 100)}
""".trimIndent()
        )
        CountDownLatch(1).await()
    }

    @Test
    fun `list test`() {
        val arr_1 = IntArray(5)
        println(arr_1.size)
        arr_1[3] = 2
        val arr_2 = Array(4) { DoubleArray(2) }
        println(arr_2.contentDeepToString())
        val zeros: NDArray<Double, D2> = mk.zeros<Double>(4, 2)
        println(zeros)
        println("zeros[3,1]->"+zeros[3,1])
        println("zeros[1] ->")
        println(zeros[1])
        val d2Array: NDArray<Double, D2> = zeros.asD2Array()
        val n = mk.ndarray(mk[mk[1, 2], mk[3, 4], mk[5, 6], mk[7, 8]])
        println(n)
        println("select all elements at")
        println("n[0..1..1]")
        println(n[0..1..1])
        println("n[2]")
        println(n[2])
        println("n[0..4,1]")
        println(n[0..4, 0])
        println(n.size)
        println("n.shape")
        val shape = n.shape
        println(shape.contentToString())
    }


    @Test
    fun view() {
        // plot(train_X)
        println("绘图结束")
        CountDownLatch(1).await()
    }

    fun plot(train_X: NDArray<Double, D2>) {
        println(train_X[0..train_X.shape[0], 0])
        println(train_X[0..train_X.shape[0], 1])

        val rand = java.util.Random()
        val n = 1000
        val data = mapOf<String, Any>(
            "x" to List(n) { rand.nextGaussian() },
            "train_x" to train_X[0..train_X.shape[0], 0].toList(),
            "train_y" to train_X[0..train_X.shape[0], 1].toList()
        )
        val plots = mapOf(
            "Density" to letsPlot(data) + geomPoint(
                color = "dark-green",
                fill = "green",
                alpha = .3,
                size = 2.0
            ) {
                x = "train_x"
                y = "train_y"
            },
            "Count" to letsPlot(data) + geomHistogram(
                color = "dark-green",
                fill = "green",
                alpha = .3,
                size = 2.0
            ) { x = "x" },

            )
        PlotUtil.plot(plots)
    }
}

