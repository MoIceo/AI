package com.example.driveehackathon

import java.time.LocalDateTime

data class TaxiOrder(
    val pickupInMeters: Int = 0,
    val pickupInSeconds: Int = 0,
    val distanceInMeters: Int = 0,
    val durationInSeconds: Int = 0,
    val orderTimestamp: LocalDateTime = LocalDateTime.now(),
    val tenderTimestamp: LocalDateTime = LocalDateTime.now(),
    val priceStartLocal: Double = 0.0,
    val priceBidLocal: Double = 0.0
)
