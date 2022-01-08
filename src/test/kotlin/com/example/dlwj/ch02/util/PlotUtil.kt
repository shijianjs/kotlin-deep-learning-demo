package com.example.dlwj.ch02.util

import javafx.application.Platform
import jetbrains.datalore.base.registration.Disposable
import jetbrains.datalore.plot.MonolithicCommon
import jetbrains.datalore.vis.swing.jfx.DefaultPlotPanelJfx
import jetbrains.letsPlot.intern.Plot
import jetbrains.letsPlot.intern.toSpec
import java.awt.Dimension
import java.awt.GridLayout
import javax.swing.*

object PlotUtil {

    fun plot(plots: Map<String, Plot>) {
        val selectedPlotKey = plots.keys.first()
        val controller = Controller(
            plots,
            selectedPlotKey,
            false
        )

        val window = JFrame("PlotUtil ${plots.keys}")
        window.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
        window.contentPane.layout = BoxLayout(window.contentPane, BoxLayout.Y_AXIS)

        // Add controls
        val controlsPanel = Box.createHorizontalBox().apply {
            // Plot selector
            val plotButtonGroup = ButtonGroup()
            for (key in plots.keys) {
                plotButtonGroup.add(
                    JRadioButton(key, key == selectedPlotKey).apply {
                        addActionListener {
                            controller.plotKey = this.text
                        }
                    }
                )
            }

            this.add(Box.createHorizontalBox().apply {
                border = BorderFactory.createTitledBorder("绘图")
                for (elem in plotButtonGroup.elements) {
                    add(elem)
                }
            })

            // Preserve aspect ratio selector
            val aspectRadioButtonGroup = ButtonGroup()
            aspectRadioButtonGroup.add(JRadioButton("原尺寸", false).apply {
                addActionListener {
                    controller.preserveAspectRadio = true
                }
            })
            aspectRadioButtonGroup.add(JRadioButton("适应窗口", true).apply {
                addActionListener {
                    controller.preserveAspectRadio = false
                }
            })

            this.add(Box.createHorizontalBox().apply {
                border = BorderFactory.createTitledBorder("纵横比")
                for (elem in aspectRadioButtonGroup.elements) {
                    add(elem)
                }
            })
        }
        window.contentPane.add(controlsPanel)

        // Add plot panel
        val plotContainerPanel = JPanel(GridLayout())
        window.contentPane.add(plotContainerPanel)

        controller.plotContainerPanel = plotContainerPanel
        controller.rebuildPlotComponent()

        SwingUtilities.invokeLater {
            window.pack()
            window.size = Dimension(800, 600)
            window.setLocationRelativeTo(null)
            window.isVisible = true
        }
    }

    private class Controller(
        private val plots: Map<String, Plot>,
        initialPlotKey: String,
        initialPreserveAspectRadio: Boolean
    ) {
        var plotContainerPanel: JPanel? = null
        var plotKey: String = initialPlotKey
            set(value) {
                field = value
                rebuildPlotComponent()
            }
        var preserveAspectRadio: Boolean = initialPreserveAspectRadio
            set(value) {
                field = value
                rebuildPlotComponent()
            }

        fun rebuildPlotComponent() {
            plotContainerPanel?.let {
                val container = plotContainerPanel!!
                // cleanup
                for (component in container.components) {
                    if (component is Disposable) {
                        component.dispose()
                    }
                }
                container.removeAll()

                // build
                container.add(createPlotPanel())
                container.parent?.revalidate()
            }
        }

        fun createPlotPanel(): JPanel {
            // Make sure JavaFX event thread won't get killed after JFXPanel is destroyed.
            Platform.setImplicitExit(false)

            val rawSpec = plots[plotKey]!!.toSpec()
            val processedSpec = MonolithicCommon.processRawSpecs(rawSpec, frontendOnly = false)

            return DefaultPlotPanelJfx(
                processedSpec = processedSpec,
                preserveAspectRatio = preserveAspectRadio,
                preferredSizeFromPlot = false,
                repaintDelay = 10,
            ) { messages ->
                for (message in messages) {
                    println("[Example App] $message")
                }
            }
        }
    }
}