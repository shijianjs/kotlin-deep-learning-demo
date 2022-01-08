package com.example.dlwj.ch02.util

object NumberUtil {
    fun Double.percent(): String {
        return "%.1f %%".format(this*100)
    }
}