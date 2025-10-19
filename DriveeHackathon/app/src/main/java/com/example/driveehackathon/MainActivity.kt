package com.example.driveehackathon

import FileProcessor
import android.graphics.drawable.Icon
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import com.example.driveehackathon.ui.theme.DriveeGray
import com.example.driveehackathon.ui.theme.DriveeGreen
import com.example.driveehackathon.ui.theme.DriveeHackathonTheme
import com.example.driveehackathon.ui.theme.DriveeLightGray
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val fileProcessor = FileProcessor(this)
        enableEdgeToEdge()
        setContent {
            DriveeHackathonTheme {
                DriveeApp(fileProcessor = fileProcessor)
            }
        }
    }
}

@Composable
fun OrderInputForm(
    taxiOrder: TaxiOrder,
    onOrderChange: (TaxiOrder) -> Unit,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        Text(
            text = "Детали поездки",
            style = MaterialTheme.typography.headlineMedium,
            modifier = Modifier.padding(bottom = 16.dp)
        )

        Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            NumberInputField(
                value = taxiOrder.pickupInMeters.toString(),
                onValueChange = {
                    onOrderChange(taxiOrder.copy(pickupInMeters = it.toIntOrNull() ?: 0))
                },
                label = "Дистанция до точки А (m)",
                modifier = Modifier.weight(1f)
            )

            NumberInputField(
                value = taxiOrder.pickupInSeconds.toString(),
                onValueChange = {
                    onOrderChange(taxiOrder.copy(pickupInSeconds = it.toIntOrNull() ?: 0))
                },
                label = "Время до точки А (с)",
                modifier = Modifier.weight(1f)
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            NumberInputField(
                value = taxiOrder.distanceInMeters.toString(),
                onValueChange = {
                    onOrderChange(taxiOrder.copy(distanceInMeters = it.toIntOrNull() ?: 0))
                },
                label = "Дистанция поездки (m)",
                modifier = Modifier.weight(1f)
            )

            NumberInputField(
                value = taxiOrder.durationInSeconds.toString(),
                onValueChange = {
                    onOrderChange(taxiOrder.copy(durationInSeconds = it.toIntOrNull() ?: 0))
                },
                label = "Длительность поездки (с)",
                modifier = Modifier.weight(1f)
            )
        }

        Spacer(modifier = Modifier.height(24.dp))

        Text(
            text = "Цены",
            style = MaterialTheme.typography.headlineMedium,
            modifier = Modifier.padding(bottom = 16.dp)
        )

        Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            NumberInputField(
                value = taxiOrder.priceStartLocal.toString(),
                onValueChange = {
                    onOrderChange(taxiOrder.copy(priceStartLocal = it.toDoubleOrNull() ?: 0.0))
                },
                label = "Начальная цена",
                modifier = Modifier.weight(1f),
                isDecimal = true
            )
        }
    }
}

@Composable
fun NumberInputField(
    value: String,
    onValueChange: (String) -> Unit,
    label: String,
    modifier: Modifier = Modifier,
    isDecimal: Boolean = false
) {
    OutlinedTextField(
        value = value,
        onValueChange = { newValue ->
            if (isDecimal) {
                if (newValue.isEmpty() || newValue.toDoubleOrNull() != null) {
                    onValueChange(newValue)
                }
            } else {
                if (newValue.isEmpty() || newValue.toIntOrNull() != null) {
                    onValueChange(newValue)
                }
            }
        },
        label = { Text(text = label) },
        modifier = modifier,
        singleLine = true,
        colors = TextFieldDefaults.colors(
            focusedContainerColor = DriveeGray,
            unfocusedContainerColor = DriveeGray,
            disabledContainerColor = DriveeGray,
            focusedTextColor = Color.White,
            unfocusedTextColor = Color.White,
            focusedLabelColor = DriveeGreen,
            unfocusedLabelColor = DriveeLightGray,
            focusedIndicatorColor = DriveeGreen,
            unfocusedIndicatorColor = DriveeLightGray
        )
    )
}

@Composable
fun DriveeApp(fileProcessor: FileProcessor? = null) {
    DriveeHackathonTheme {
        var taxiOrder by remember { mutableStateOf(TaxiOrder()) }
        var aiPredictions by remember { mutableStateOf<AIPricePrediction?>(null) }
        var isLoading by remember { mutableStateOf(false) }
        var errorMessage by remember { mutableStateOf<String?>(null) }
        var successMessage by remember { mutableStateOf<String?>(null) }
        val coroutineScope = rememberCoroutineScope()

        fun processOrderAndGetPredictions() {
            isLoading = true
            errorMessage = null
            successMessage = null

            coroutineScope.launch {
                try {
                    // 1. Записываем данные в CSV
                    val csvSuccess = fileProcessor?.writeToCsv(taxiOrder) ?: false
                    if (!csvSuccess) {
                        errorMessage = "Ошибка записи в CSV файл"
                        return@launch
                    }

                    successMessage = "Данные записаны в CSV"

                    // 2. Запускаем Python скрипт
                    val pythonSuccess = fileProcessor?.runPythonScript() ?: false
                    if (!pythonSuccess) {
                        errorMessage = "Ошибка выполнения Python скрипта"
                        return@launch
                    }

                    successMessage = "Python скрипт выполнен успешно"

                    // 3. Читаем предсказания из JSON
                    val predictions = fileProcessor?.readPredictions() ?: emptyList()

                    aiPredictions = AIPricePrediction(prices = predictions)
                    successMessage = "Предсказания получены успешно"

                } catch (e: Exception) {
                    errorMessage = "Ошибка: ${e.message}"
                } finally {
                    isLoading = false
                }
            }
        }

        fun clearMessages() {
            errorMessage = null
            successMessage = null
        }

        Surface(
            modifier = Modifier.fillMaxSize(),
            color = MaterialTheme.colorScheme.background
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .padding(16.dp)
            ) {
                Text(
                    text = "Drivee AI ценообразование",
                    style = MaterialTheme.typography.headlineLarge,
                    color = DriveeGreen,
                    modifier = Modifier.padding(bottom = 24.dp)
                )

                OrderInputForm(
                    taxiOrder = taxiOrder,
                    onOrderChange = { taxiOrder = it },
                    modifier = Modifier.fillMaxWidth()
                )

                Spacer(modifier = Modifier.height(24.dp))

                successMessage?.let { message ->
                    SuccessMessage(
                        message = message,
                        onDismiss = { clearMessages() },
                        modifier = Modifier.fillMaxWidth()
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                }

                errorMessage?.let { message ->
                    ErrorMessage(
                        message = message,
                        onDismiss = { clearMessages() },
                        modifier = Modifier.fillMaxWidth()
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                }

                Button(
                    onClick = {
                        if (fileProcessor != null) {
                            processOrderAndGetPredictions()
                        }
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(56.dp),
                    enabled = !isLoading && isValidOrder(taxiOrder),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = DriveeGreen,
                        contentColor = Color.Black
                    )
                ) {
                    if (isLoading) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(20.dp),
                            color = Color.Black,
                            strokeWidth = 2.dp
                        )
                    } else {
                        Text(
                            text = "Get AI Price Recommendations",
                            style = MaterialTheme.typography.labelLarge
                        )
                    }
                }

                Spacer(modifier = Modifier.height(32.dp))

                aiPredictions?.let { predictions ->
                    PriceRecommendationsSection(
                        predictions = predictions,
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            }
        }
    }
}

private fun isValidOrder(order: TaxiOrder): Boolean {
    return order.distanceInMeters > 0 &&
            order.durationInSeconds > 0 &&
            order.priceStartLocal > 0
}

@Composable
fun PriceRecommendationsSection(
    predictions: AIPricePrediction,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        Text(
            text = "AI Price Recommendations",
            style = MaterialTheme.typography.headlineMedium,
            modifier = Modifier.padding(bottom = 16.dp)
        )

        // Price cards
        predictions.prices.forEach { prediction ->
            PriceCard(
                prediction = prediction,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 8.dp)
            )
        }
    }
}

@Composable
fun PriceCard(
    prediction: PricePrediction,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = if (prediction.recommended) {
                DriveeGreen.copy(alpha = 0.2f)
            } else {
                DriveeGray
            }
        ),
        elevation = CardDefaults.cardElevation(
            defaultElevation = if (prediction.recommended) 8.dp else 4.dp
        ),
        border = if (prediction.recommended) {
            BorderStroke(2.dp, DriveeGreen)
        } else {
            null
        }
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(
                modifier = Modifier.weight(1f)
            ) {
                // Заголовок с рекомендованной меткой
                Row(
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Price Offer",
                        style = MaterialTheme.typography.bodyLarge,
                        color = Color.White
                    )
                    if (prediction.recommended) {
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "RECOMMENDED",
                            style = MaterialTheme.typography.labelSmall,
                            color = DriveeGreen,
                            modifier = Modifier
                                .background(
                                    color = DriveeGreen.copy(alpha = 0.2f),
                                    shape = RoundedCornerShape(4.dp)
                                )
                                .padding(horizontal = 6.dp, vertical = 2.dp)
                        )
                    }
                }

                Text(
                    text = "Success probability: ${(prediction.successProbability * 100).toInt()}%",
                    style = MaterialTheme.typography.bodyMedium,
                    color = DriveeLightGray,
                    modifier = Modifier.padding(top = 4.dp)
                )

                LinearProgressIndicator(
                    progress = prediction.successProbability.toFloat(),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 8.dp)
                        .height(4.dp),
                    color = when {
                        prediction.successProbability > 0.7 -> DriveeGreen
                        prediction.successProbability > 0.4 -> Color.Yellow
                        else -> Color.Red
                    },
                    trackColor = DriveeLightGray.copy(alpha = 0.3f)
                )
            }

            Column(
                horizontalAlignment = Alignment.End
            ) {
                Text(
                    text = "%.0f".format(prediction.amount),
                    style = MaterialTheme.typography.headlineMedium,
                    color = DriveeGreen
                )
                Text(
                    text = "RUB",
                    style = MaterialTheme.typography.bodySmall,
                    color = DriveeLightGray
                )
            }
        }
    }
}

@Composable
fun SuccessMessage(
    message: String,
    onDismiss: () -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = DriveeGreen.copy(alpha = 0.1f)
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = message,
                style = MaterialTheme.typography.bodyMedium,
                color = DriveeGreen,
                modifier = Modifier.weight(1f)
            )
            IconButton(onClick = onDismiss) {
                Icon(
                    imageVector = Icons.Default.Close,
                    contentDescription = "Close",
                    tint = DriveeGreen
                )
            }
        }
    }
}

@Composable
fun ErrorMessage(
    message: String,
    onDismiss: () -> Unit,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = Color.Red.copy(alpha = 0.1f)
        )
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = message,
                style = MaterialTheme.typography.bodyMedium,
                color = Color.Red,
                modifier = Modifier.weight(1f)
            )
            IconButton(onClick = onDismiss) {
                Icon(
                    imageVector = Icons.Default.Close,
                    contentDescription = "Close",
                    tint = Color.Red
                )
            }
        }
    }
}