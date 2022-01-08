package com.example.dlwj.ch02.util

import kotlin.math.*
import kotlin.random.Random

/**
 * 高斯分布
 *
 * 正态分布
 * @param mean 平均数
 * @param variance 方差
 */
class GaussianDistribution(
    val mean: Double,
    val variance: Double,
    val rng: Random = Random.Default
) {

    init {
        require(variance >= 0) { "Variance must be nog-negative value." }
    }

    fun random(): Double {
        var r = 0.0
        while (r == 0.0) {
            r = rng.nextDouble()
        }
        val c = sqrt(-2.0 * ln(r))
        return if (rng.nextDouble() < 0.5) {
            c * sin(2.0 * PI * rng.nextDouble()) * variance + mean
        } else {
            c * cos(2.0 * PI * rng.nextDouble()) * variance + mean
        }
    }
}