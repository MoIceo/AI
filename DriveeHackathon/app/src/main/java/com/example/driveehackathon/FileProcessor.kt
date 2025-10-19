import android.content.Context
import com.example.driveehackathon.PricePrediction
import com.example.driveehackathon.TaxiOrder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.*
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import org.json.JSONArray
import org.json.JSONObject

class FileProcessor(private val context: Context) {

    private fun getAppDataDirectory(): File {
        return File(context.filesDir, "data").apply {
            if (!exists()) {
                mkdirs() // Создаем папку если ее нет
            }
        }
    }
    suspend fun copyDatasetFromAssets(): Boolean = withContext(Dispatchers.IO) {
        try {
            val dataDir = getAppDataDirectory()
            val outputFile = File(dataDir, "dataset.csv")

            // Копируем из assets во внутреннее хранилище
            val inputStream = context.assets.open("dataset.csv")
            FileOutputStream(outputFile).use { output ->
                inputStream.copyTo(output)
            }
            inputStream.close()
            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }

    suspend fun writeToCsv(taxiOrder: TaxiOrder): Boolean = withContext(Dispatchers.IO) {
        try {
            val dataDir = getAppDataDirectory()
            val file = File(dataDir, "data.csv")

            val fileExists = file.exists()

            FileWriter(file, true).use { writer ->
                if (!fileExists) {
                    writer.write("pickup_in_meters;pickup_in_seconds;distance_in_meters;duration_in_seconds;order_timestamp;tender_timestamp;price_start_local\n")
                }

                writer.write(
                    "${taxiOrder.pickupInMeters};" +
                            "${taxiOrder.pickupInSeconds};" +
                            "${taxiOrder.distanceInMeters};" +
                            "${taxiOrder.durationInSeconds};" +
                            "${taxiOrder.orderTimestamp.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)};" +
                            "${taxiOrder.tenderTimestamp.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)};" +
                            "${taxiOrder.priceStartLocal}\n"
                )
            }
            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }

    suspend fun runPythonScript(): Boolean = withContext(Dispatchers.IO) {
        try {
            val dataDir = getAppDataDirectory()

            // Копируем Python скрипт из assets если его нет
            val pythonScript = File(dataDir, "predictor.py")
            if (!pythonScript.exists()) {
                copyPythonScriptFromAssets(pythonScript)
            }

            // Создаем копию dataset.csv во внутренней директории если нужно
            val datasetFile = File(dataDir, "dataset.csv")
            if (!datasetFile.exists()) {
                copyDatasetFromAssets(datasetFile)
            }

            // Запускаем Python скрипт
            val process = ProcessBuilder()
                .command("python", pythonScript.absolutePath)
                .directory(dataDir)
                .start()

            val exitCode = process.waitFor()
            exitCode == 0
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }

    private suspend fun copyDatasetFromAssets(destination: File) = withContext(Dispatchers.IO) {
        try {
            val inputStream = context.assets.open("dataset.csv")
            FileOutputStream(destination).use { output ->
                inputStream.copyTo(output)
            }
            inputStream.close()
        } catch (e: Exception) {
            // Если dataset.csv нет в assets, создаем пустой файл
            destination.createNewFile()
        }
    }

    private suspend fun copyPythonScriptFromAssets(destination: File) = withContext(Dispatchers.IO) {
        try {
            val inputStream = context.assets.open("predictor.py")
            FileOutputStream(destination).use { output ->
                inputStream.copyTo(output)
            }
            inputStream.close()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    suspend fun readPredictions(): List<PricePrediction> = withContext(Dispatchers.IO) {
        try {
            val dataDir = getAppDataDirectory()
            val file = File(dataDir, "prediction.json")

            if (!file.exists()) {
                return@withContext emptyList()
            }

            val jsonString = file.readText()
            parsePricePredictions(jsonString)
        } catch (e: Exception) {
            e.printStackTrace()
            emptyList()
        }
    }

    private fun parsePricePredictions(jsonString: String): List<PricePrediction> {
        return try {
            val jsonObject = JSONObject(jsonString)
            val pricesArray = jsonObject.getJSONArray("prices")
            val predictions = mutableListOf<PricePrediction>()

            for (i in 0 until pricesArray.length()) {
                val priceObj = pricesArray.getJSONObject(i)
                predictions.add(
                    PricePrediction(
                        amount = priceObj.getDouble("amount"),
                        successProbability = priceObj.getDouble("success_probability"),
                        recommended = priceObj.getBoolean("recommended")
                    )
                )
            }
            predictions
        } catch (e: Exception) {
            listOf(
                PricePrediction(1000.0, 0.3, false),
                PricePrediction(1500.0, 0.5, true),
                PricePrediction(2000.0, 0.2, false)
            )
        }
    }
}