package com.example.driveehackathon.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

val DriveeBlack = Color(0xFF121212)
val DriveeGreen = Color(0xFF00D455)
val DriveeDarkGreen = Color(0xFF00B248)
val DriveeGray = Color(0xFF2A2A2A)
val DriveeLightGray = Color(0xFF424242)

private val DriveeColorScheme = darkColorScheme(
    primary = DriveeGreen,
    onPrimary = Color.Black,
    background = DriveeBlack,
    surface = DriveeGray,
    onBackground = Color.White,
    onSurface = Color.White
)

@Composable
fun DriveeHackathonTheme(
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = DriveeColorScheme,
        typography = Typography,
        content = content
    )
}