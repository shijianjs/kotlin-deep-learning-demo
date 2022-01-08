package com.example.dlwj.ch02.single

import com.example.dlwj.ch02.util.step
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.get

/**
 * 单层感知器
 *
 * @param w 权重
 */
class Perceptrons(var nIn: Int, var w: DoubleArray = DoubleArray(nIn)) {


    /**
     * 训练
     *
     * @param input 二维坐标，输入
     * @param t 输出，只有 1，-1两种结果
     * @param learningRate 学习速率
     */
    fun train(input: MultiArray<Double, D1>, t: Int, learningRate: Double): Int {
        var classified = 0
        val e_f_input: Double = (0 until nIn).sumOf { w[it] * input[it] * t }
        if (e_f_input > 0) {
            classified = 1
        } else {
            (0 until nIn).forEach { w[it] += learningRate * input[it] * t }
            println("权重："+w.contentToString())
        }
        return classified
    }

    fun predict(x: MultiArray<Double, D1>): Int {
        val preActivation = (0 until nIn).sumOf { w[it] * x[it] }
        return step(preActivation)
    }


}
