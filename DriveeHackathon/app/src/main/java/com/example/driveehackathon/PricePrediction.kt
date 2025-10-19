package com.example.driveehackathon

data class PricePrediction(
    val amount: Double,
    val successProbability: Double,
    val recommended: Boolean
)

data class AIPricePrediction(
    val prices: List<PricePrediction> = emptyList()
)