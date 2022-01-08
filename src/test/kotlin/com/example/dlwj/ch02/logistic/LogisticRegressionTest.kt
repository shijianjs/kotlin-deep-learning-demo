package com.example.dlwj.ch02.logistic

import com.example.dlwj.ch02.util.GaussianDistribution
import com.example.dlwj.ch02.util.NumberUtil.percent
import com.example.dlwj.ch02.util.PlotUtil
import jetbrains.letsPlot.geom.geomHistogram
import jetbrains.letsPlot.geom.geomPoint
import jetbrains.letsPlot.letsPlot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.indexOf
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import org.jetbrains.kotlinx.multik.ndarray.operations.toListD2
import org.junit.jupiter.api.Test
import java.util.concurrent.CountDownLatch
import kotlin.concurrent.thread
import kotlin.random.Random

class LogisticRegressionTest {

    @Test
    fun main() {
        val rng = Random(1234)
        val patterns = 3
        val train_N = 400 * patterns
        val test_N = 60 * patterns
        val nIn = 2
        val nOut = patterns


        val train_X = mk.zeros<Double>(train_N, nIn)
        val train_T = mk.zeros<Int>(train_N, nOut)
        val test_X = mk.zeros<Double>(test_N, nIn)
        val test_T = mk.zeros<Int>(test_N, nOut)
        val predicted_T = mk.zeros<Int>(test_N, nOut)

        val epochs = 2000
        var learningRate = 0.2

        val minibatchSize = 50
        val minibatch_N = train_N / minibatchSize

        val train_X_minibatch = mk.zeros<Double>(minibatch_N, minibatchSize, nIn)
        val train_T_minibatch = mk.zeros<Int>(minibatch_N, minibatchSize, nOut)
        val minibatchIndex = (0 until train_N).toMutableList()
        minibatchIndex.shuffle()

        val g1 = GaussianDistribution(-2.0, 1.0, rng)
        val g2 = GaussianDistribution(2.0, 1.0, rng)
        val g3 = GaussianDistribution(0.0, 1.0, rng)

        // class1 : x1 ~ N(-2.0, 1.0), y1 ~ N(+2.0, 1.0)
        for (i in 0 until (train_N / patterns - 1)) {
            train_X[i, 0] = g1.random()
            train_X[i, 1] = g2.random()
            train_T[i] = mk.ndarray(mk[1, 0, 0])
        }
        for (i in 0 until (test_N / patterns - 1)) {
            test_X[i, 0] = g1.random()
            test_X[i, 1] = g2.random()
            test_T[i] = mk.ndarray(mk[1, 0, 0])
        }

        // class2 : x1 ~ N(+2.0, 1.0), y1 ~ N(-2.0, 1.0)
        for (i in (train_N / patterns - 1) until (train_N / patterns * 2 - 1)) {
            train_X[i, 0] = g2.random()
            train_X[i, 1] = g1.random()
            train_T[i] = mk.ndarray(mk[0, 1, 0])
        }
        for (i in (test_N / patterns - 1) until (test_N / patterns * 2 - 1)) {
            test_X[i, 0] = g2.random()
            test_X[i, 1] = g1.random()
            test_T[i] = mk.ndarray(mk[0, 1, 0])
        }

        // class3 : x1 ~ N(0.0, 1.0), y1 ~ N(0.0, 1.0)
        for (i in (train_N / patterns * 2 - 1) until train_N) {
            train_X[i, 0] = g3.random()
            train_X[i, 1] = g3.random()
            train_T[i] = mk.ndarray(mk[0, 0, 1])
        }
        for (i in (test_N / patterns * 2 - 1) until test_N) {
            test_X[i, 0] = g3.random()
            test_X[i, 1] = g3.random()
            test_T[i] = mk.ndarray(mk[0, 0, 1])
        }

        val data = mapOf<String, Any>(
            "train_x" to train_X[0..train_X.shape[0], 0].toList(),
            "train_y" to train_X[0..train_X.shape[0], 1].toList(),
            "type" to train_T.toListD2().map {
                when (it.indexOf(1)) {
                    0 -> "green"
                    1 -> "red"
                    2 -> "blue"
                    else -> "black"
                }
            }
        )
        val plots = mapOf(
            "Density" to letsPlot(data) + geomPoint(
                // color = "dark-green",
                // fill = "green",
                alpha = .3,
                size = 2.0
            ) {
                x = "train_x"
                y = "train_y"
                color = "type"
                fill="type"
            },
            "Count" to letsPlot(data) + geomHistogram(
                color = "dark-green",
                fill = "green",
                alpha = .3,
                size = 2.0
            ) { x = "train_x" },

            )
        PlotUtil.plot(plots)

        // 分批
        for (i in 0 until minibatch_N) {
            for (j in 0 until minibatchSize) {
                val index = minibatchIndex[i * minibatchSize + j]
                train_X_minibatch[i, j] = train_X[index]
                train_T_minibatch[i, j] = train_T[index]
            }
        }

        // 分类器
        val classifier = LogisticRegression(nIn, nOut)

        // 训练
        for (epoch in 0 until epochs) {
            for (batch in 0 until minibatch_N) {
                classifier.train(train_X_minibatch[batch], train_T_minibatch[batch], minibatchSize, learningRate)
            }
            learningRate *= 0.95
        }

        //     测试
        for (i in 0 until test_N) {
            predicted_T[i] = classifier.predict(test_X[i])
        }

        // 结果评估
        val confusionMatrix: NDArray<Int, D2> = mk.zeros<Int>(patterns, patterns)
        var accuracy = 0.0
        val precision: NDArray<Double, D1> = mk.zeros<Double>(patterns)
        val recall = mk.zeros<Double>(patterns)

        for (i in 0 until test_N) {
            val predicted_ = predicted_T[i].indexOf(1)
            val actual_ = test_T[i].indexOf(1)
            confusionMatrix[actual_, predicted_] += 1
        }

        for (i in 0 until patterns) {
            var col_ = 0.0
            var row_ = 0.0
            for (j in 0 until patterns) {
                if (i == j) {
                    accuracy += confusionMatrix[i, j]
                    precision[i] += confusionMatrix[j, i].toDouble()
                    recall[i] += confusionMatrix[i, j].toDouble()
                }
                col_ += confusionMatrix[j, i]
                row_ += confusionMatrix[i, j]
            }
            precision[i] /= col_
            recall[i] /= row_
        }
        accuracy /= test_N
        println(
            """
            ---------------------------------
                     逻辑回归模型评估
            ---------------------------------
            准确率：${accuracy.percent()}
            精确度：${precision.toList().map { it.percent() }}
            召回率：${recall.toList().map { it.percent() }}
        """.trimIndent()
        )

        CountDownLatch(1).await()
    }

    init {
        // 正常main方法启动，非守护现场不停止就不退出jvm，这条规则才有效
        // thread { CountDownLatch(1).await() }
    }

    companion object{
        @JvmStatic
        fun main(args: Array<String>) {
            LogisticRegressionTest().main()
        }
    }
}