package com.example.dlwj.ch02.util

import kotlin.random.Random


object RandomGenerator {


    fun uniform(min: Double, max: Double, rng: Random): Double {
        return rng.nextDouble(min, max)
    }
}