package com.example.driveehackathon

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.example.driveehackathon.ui.theme.DriveeGray
import com.example.driveehackathon.ui.theme.DriveeGreen
import com.example.driveehackathon.ui.theme.DriveeHackathonTheme
import com.example.driveehackathon.ui.theme.DriveeLightGray
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            DriveeHackathonTheme {
                DriveeApp()
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

            //NumberInputField(
            //    value = taxiOrder.priceBidLocal.toString(),
            //    onValueChange = {
            //        onOrderChange(taxiOrder.copy(priceBidLocal = it.toDoubleOrNull() ?: 0.0))
            //    },
            //    label = "AI Bid Price",
            //    modifier = Modifier.weight(1f),
            //    isDecimal = true
            //)
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
fun DriveeApp() {
    DriveeHackathonTheme {
        var taxiOrder by remember { mutableStateOf(TaxiOrder()) }
        var aiPredictions by remember { mutableStateOf<AIPricePrediction?>(null) }
        var isLoading by remember { mutableStateOf(false) }

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

                Button(
                    onClick = {
                        isLoading = true
                        simulateAIProcessing(taxiOrder) { predictions ->
                            aiPredictions = predictions
                            isLoading = false
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

private fun simulateAIProcessing(
    order: TaxiOrder,
    onComplete: (AIPricePrediction) -> Unit
) {
    kotlinx.coroutines.MainScope().launch {
        kotlinx.coroutines.delay(2000)

        val basePrice = order.priceStartLocal
        val recommendedPrices = listOf(
            basePrice * 0.9,
            basePrice * 1.1,
            basePrice * 1.25
        ).map { it.coerceAtLeast(order.priceBidLocal) }

        onComplete(AIPricePrediction(recommendedPrices, 0.85))
    }
}

@Composable
fun PriceRecommendationsSection(
    predictions: AIPricePrediction,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        Text(
            text = "Рекомендации нейросети",
            style = MaterialTheme.typography.headlineMedium,
            modifier = Modifier.padding(bottom = 16.dp)
        )

        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(bottom = 16.dp)
        ) {
            Text(
                text = "Коэффициент: ",
                style = MaterialTheme.typography.bodyMedium,
                color = DriveeLightGray
            )
            Text(
                text = "${(predictions.confidence * 100).toInt()}%",
                style = MaterialTheme.typography.bodyMedium,
                color = DriveeGreen
            )
        }

        predictions.prices.forEachIndexed { index, price ->
            PriceCard(
                price = price,
                type = when (index) {
                    0 -> "Эконом"
                    1 -> "Обычный"
                    else -> "Бизнес"
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 8.dp)
            )
        }
    }
}

@Composable
fun PriceCard(
    price: Double,
    type: String,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = DriveeGray
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    text = type,
                    style = MaterialTheme.typography.bodyLarge,
                    color = Color.White
                )
                Text(
                    text = "AI Recommended",
                    style = MaterialTheme.typography.bodyMedium,
                    color = DriveeLightGray
                )
            }

            Text(
                text = "%.2f".format(price),
                style = MaterialTheme.typography.headlineMedium,
                color = DriveeGreen
            )
        }
    }
}