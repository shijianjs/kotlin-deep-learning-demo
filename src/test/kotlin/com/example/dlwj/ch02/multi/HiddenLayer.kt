package com.example.dlwj.ch02.multi

import com.example.dlwj.ch02.util.RandomGenerator.uniform
import com.example.dlwj.ch02.util.dsigmoid
import com.example.dlwj.ch02.util.dtanh
import com.example.dlwj.ch02.util.sigmoid
import com.example.dlwj.ch02.util.tanh
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.random.Random

class HiddenLayer(
    val nIn: Int,
    val nOut: Int,
    WIn: D2Array<Double>? = null,
    val b: D1Array<Double> = mk.zeros(nOut),
    val rng: Random = Random(1234),
    activationName: String? = null,
) {
    val W: D2Array<Double>
    val activation: (Double) -> Double
    val dactivation: (Double) -> Double

    init {
        if (WIn == null) {
            W = mk.zeros<Double>(nOut, nIn)
            val w_ = 1.0 / nIn
            (0 until nOut).forEach { j ->
                (0 until nIn).forEach { i ->
                    W[j, i] = uniform(-w_, w_, rng)
                }
            }
        } else {
            W = WIn
        }
        when (activationName) {
            "sigmoid", null -> {
                this.activation = ::sigmoid
                this.dactivation = ::dsigmoid
            }
            "tanh" -> {
                this.activation = ::tanh
                this.dactivation = ::dtanh
            }
            else -> throw IllegalArgumentException("activation function not supported")
        }


    }

    fun output(x: MultiArray<Double, D1>): D1Array<Double> {
        val y = mk.zeros<Double>(nOut)
        (0 until nOut).forEach { j ->
            y[j] = activation((0 until nIn).sumOf { i -> W[j, i] * x[i] } + b[j])
        }
        return y
    }

    fun forward(x: MultiArray<Double, D1>): D1Array<Double> {
        return output(x)
    }

    fun backward(
        X: MultiArray<Double, D2>,
        Z: D2Array<Double>,
        dY: D2Array<Double>,
        Wprev: D2Array<Double>,
        minibatchSize: Int,
        learningRate: Double
    ): D2Array<Double> {
        val dZ = mk.zeros<Double>(minibatchSize, nOut)
        val grad_W = mk.zeros<Double>(nOut, nIn)
        val grad_b = mk.zeros<Double>(nOut)
        for (n in 0 until minibatchSize) {
            for (j in 0 until nOut) {
                for (k in 0 until dY[0].size) {
                    dZ[n, j] += Wprev[k, j] * dY[n, k]
                }
                dZ[n, j] *= dactivation(Z[n, j])
                for (i in 0 until nIn) {
                    grad_W[j, i] += dZ[n][j] * X[n, i]
                }
                grad_b[j] += dZ[n, j]
            }
        }
        for (j in 0 until nOut) {
            for (i in 0 until nIn) {
                W[j, i] -= learningRate * grad_W[j, i] / minibatchSize
            }
            b[j] -= learningRate * grad_b[j] / minibatchSize
        }
        return dZ
    }

}