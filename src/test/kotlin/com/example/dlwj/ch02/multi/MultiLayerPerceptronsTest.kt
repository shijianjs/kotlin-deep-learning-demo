package com.example.dlwj.ch02.multi

import com.example.dlwj.ch02.util.NumberUtil.percent
import com.example.dlwj.ch02.util.PlotUtil
import jetbrains.letsPlot.geom.geomHistogram
import jetbrains.letsPlot.geom.geomPoint
import jetbrains.letsPlot.letsPlot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.indexOf
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import org.jetbrains.kotlinx.multik.ndarray.operations.toListD2
import org.junit.jupiter.api.Test
import java.util.concurrent.CountDownLatch
import kotlin.random.Random
import kotlin.Int

class MultiLayerPerceptronsTest {


    @Test
    fun main() {
        val rng = Random(123)
        val patterns = 2
        val train_N = 4
        val test_N = 4
        val nIn = 2
        val nHidden = 3
        val nOut = patterns

        val predicted_T = mk.zeros<Int>(test_N, nOut)
        val epochs = 5000
        val learningRate = 0.1
        val minibatchSize = 1
        val minibatch_N = train_N / minibatchSize
        val train_X_minibatch = mk.zeros<Double>(minibatch_N, minibatchSize, nIn)
        val train_T_minibatch = mk.zeros<Int>(minibatch_N, minibatchSize, nOut)
        val minibatchIndex = (0 until train_N).toMutableList()
        minibatchIndex.shuffle()

        val train_X = mk.ndarray(mk[mk[0.0, 0.0], mk[0.0, 1.0], mk[1.0, 0.0], mk[1.0, 1.0]])
        val train_T = mk.ndarray(mk[mk[0, 1], mk[1, 0], mk[1, 0], mk[0, 1]])
        val test_X = mk.ndarray(mk[mk[0.0, 0.0], mk[0.0, 1.0], mk[1.0, 0.0], mk[1.0, 1.0]])
        val test_T = mk.ndarray(mk[mk[0, 1], mk[1, 0], mk[1, 0], mk[0, 1]])



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
                train_X_minibatch[i, j] = train_X[minibatchIndex[i * minibatchSize + j]]
                train_T_minibatch[i, j] = train_T[minibatchIndex[i * minibatchSize + j]]
            }
        }

        // 分类器
        val classifier = MultiLayerPerceptrons(nIn, nHidden, nOut, rng)
        // 训练
        for (epoch in 0 until epochs) {
            for (batch in 0 until minibatch_N) {
                classifier.train(train_X_minibatch[batch], train_T_minibatch[batch], minibatchSize, learningRate)
            }
        }
        // 测试
        for (i in 0 until test_N) {
            predicted_T[i] = classifier.predict(test_X[i])
        }

        // 评估
        val confusionMatrix = mk.zeros<Int>(patterns, patterns)
        var accuracy = 0.0
        val precision = mk.zeros<Double>(patterns)
        val recall = mk.zeros<Double>(patterns)

        for (i in 0 until test_N) {
            val prediated = predicted_T[i].indexOf(1)
            val actual = test_T[i].indexOf(1)
            confusionMatrix[actual, prediated] += 1
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
}