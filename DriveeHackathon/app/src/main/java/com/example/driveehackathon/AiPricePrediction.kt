package com.example.driveehackathon

data class AIPricePrediction(
    val prices: List<Double> = emptyList(),
    val confidence: Double = 0.0
)
