package com.example.dlwj.ch02.util

import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.slice
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.max
import org.jetbrains.kotlinx.multik.ndarray.operations.sum
import org.jetbrains.kotlinx.multik.ndarray.operations.toMutableList
import kotlin.math.E
import kotlin.math.exp
import kotlin.math.pow

fun step(x: Double): Int {
    return if (x >= 0) 1 else -1;
}

/**
 * pow 指数
 */
fun sigmoid(x: Double): Double {
    return 1.0 / (1.0 + E.pow(-x))
}

fun dsigmoid(y: Double): Double = y * (1 - y)

fun tanh(x: Double) = kotlin.math.tanh(x)

fun dtanh(y: Double) = 1 - y * y


fun softmax(x: D1Array<Double>, n: Int): D1Array<Double> {
    val sliceArray = x[0 .. n]
    val max = sliceArray.max()!!
    val y = sliceArray.map { exp(it - max) }
    val sum: Double = y.sum()
    return y.map { it / sum }
}