package com.example.dlwj.ch02.multi

import com.example.dlwj.ch02.logistic.LogisticRegression
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.random.Random

class MultiLayerPerceptrons(
    val nIn: Int,
    val nHidden: Int,
    val nOut: Int,
    val rng: Random = Random(1234),
    val hiddenLayer: HiddenLayer = HiddenLayer(nIn = nIn, nOut = nHidden, rng = rng, activationName = "tanh"),
    val logisticLayer: LogisticRegression = LogisticRegression(nIn = nHidden, nOut = nOut)
) {
    fun train(X: MultiArray<Double, D2>, T: MultiArray<Int, D2>, minibatchSize: Int, learningRate: Double) {
        val Z = mk.zeros<Double>(minibatchSize, nHidden)
        for (n in 0 until minibatchSize) {
            Z[n] = hiddenLayer.forward(X[n])
        }
        val dY = logisticLayer.train(Z, T, minibatchSize, learningRate)
        hiddenLayer.backward(X, Z, dY, logisticLayer.W, minibatchSize, learningRate)
    }

    fun predict(x: MultiArray<Double, D1>):D1Array<Int>{
        val z = hiddenLayer.output(x)
        return logisticLayer.predict(z)
    }



}