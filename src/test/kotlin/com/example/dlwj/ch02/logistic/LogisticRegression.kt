package com.example.dlwj.ch02.logistic

import com.example.dlwj.ch02.util.softmax
import org.jetbrains.kotlinx.multik.api.math.argMax
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.*

class LogisticRegression(
    val nIn: Int,
    val nOut: Int,
    val W: D2Array<Double> = mk.zeros<Double>(nOut, nIn),
    val b: D1Array<Double> = mk.zeros<Double>(nOut),
) {

    fun train(X: MultiArray<Double, D2>, T: MultiArray<Int, D2>, minibatchSize: Int, learningRate: Double): D2Array<Double> {
        val grad_W = mk.zeros<Double>(nOut, nIn)
        val grad_b = mk.zeros<Double>(nOut)
        val dY = mk.zeros<Double>(minibatchSize, nOut)
        for (n in (0 until minibatchSize)) {
            val predicted_Y_: D1Array<Double> = output(X[n])
            for (j in (0 until nOut)) {
                val d = predicted_Y_[j]
                val i1 = T[n, j]
                dY[n, j] = d - i1
                for (i in (0 until nIn)) {
                    grad_W[j, i] += dY[n, j] * X[n, i]
                }
                grad_b[j] += dY[n, j]
            }
        }
        for (j in (0 until nOut)) {
            for (i in (0 until nIn)) {
                W[j, i] -= learningRate * grad_W[j, i] / minibatchSize
            }
            b[j]-=learningRate*grad_b[j]/minibatchSize
        }
        return dY

    }

    private fun output(x: MultiArray<Double, D1>): D1Array<Double> {
        val preActivation = mk.zeros<Double>(nOut)
        for (j in (0 until nOut)) {
            for (i in (0 until nIn)) {
                preActivation[j] += W[j, i] * x[i]
            }
            preActivation[j] += b[j]
        }
        return softmax(preActivation, nOut)
    }

    fun predict(x: MultiArray<Double, D1>): D1Array<Int> {
        val y = output(x)
        val t = mk.zeros<Int>(nOut)
        val argmax = y.argMax()
        t[argmax] = 1
        return t
    }

}