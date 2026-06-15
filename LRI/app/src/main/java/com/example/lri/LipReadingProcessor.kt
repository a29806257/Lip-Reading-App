package com.example.lri

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader

class LipReadingProcessor(context: Context) {

    private val idxToLabelMap = mutableMapOf<Int, String>()

    init {
        loadLabels(context)
    }

    private fun loadLabels(context: Context) {
        try {
            val assetManager = context.assets
            val inputStream = assetManager.open("labels.json")
            val reader = BufferedReader(InputStreamReader(inputStream))
            val jsonString = reader.use { it.readText() }
            val jsonObject = JSONObject(jsonString)

            if (jsonObject.has("label2idx")) {
                val label2idx = jsonObject.getJSONObject("label2idx")
                val keys = label2idx.keys()
                while (keys.hasNext()) {
                    val label = keys.next()
                    val idx = label2idx.getInt(label)
                    idxToLabelMap[idx] = label
                }
            } else if (jsonObject.has("idx2label")) {
                val idx2label = jsonObject.getJSONObject("idx2label")
                val keys = idx2label.keys()
                while (keys.hasNext()) {
                    val idxStr = keys.next()
                    val label = idx2label.getString(idxStr)
                    idxToLabelMap[idxStr.toInt()] = label
                }
            } else {
                // 相容單純的 {"0": "a", "1": "b"} 格式
                val keys = jsonObject.keys()
                while (keys.hasNext()) {
                    val idxStr = keys.next()
                    val label = jsonObject.getString(idxStr)
                    idxToLabelMap[idxStr.toInt()] = label
                }
            }
            Log.d("LRI_DEBUG", "✅ Labels載入完成，共 ${idxToLabelMap.size} 個類別")
        } catch (e: Exception) {
            Log.e("LRI_DEBUG", "❌ 讀取 labels.json 失敗", e)
        }
    }

    fun decodeAndLog(outputProbabilities: FloatArray, wordIndex: Int): String {
        if (idxToLabelMap.isEmpty()) {
            Log.e("LRI_DEBUG", "❌ 嚴重錯誤：字典是空的，不管猜什麼都會變成 unknown")
            return "unknown"
        }

        val top3Indices = outputProbabilities.indices.sortedByDescending { outputProbabilities[it] }.take(3)

        Log.d("LRI_DEBUG", "====== 第 $wordIndex 個字的原始模型輸出 (Top 3) ======")
        for (i in top3Indices) {
            val label = idxToLabelMap[i] ?: "找不到標籤"
            val score = outputProbabilities[i]
            Log.d("LRI_DEBUG", "👉 Index: $i | 標籤(拼音): $label | 總分數: $score")
        }
        Log.d("LRI_DEBUG", "==================================================")

        val bestIdx = top3Indices[0]
        return idxToLabelMap[bestIdx] ?: "unknown"
    }

    fun decodeOutput(outputProbabilities: FloatArray): String {
        var maxIndex = -1
        var maxScore = Float.NEGATIVE_INFINITY
        for (i in outputProbabilities.indices) {
            if (outputProbabilities[i] > maxScore) {
                maxScore = outputProbabilities[i]
                maxIndex = i
            }
        }
        return if (maxIndex != -1 && idxToLabelMap.containsKey(maxIndex)) idxToLabelMap[maxIndex]!! else "unknown"
    }

    fun getTopKPinyins(outputProbabilities: FloatArray, k: Int = 3): List<String> {
        if (idxToLabelMap.isEmpty()) return listOf("unknown")

        return outputProbabilities.indices
            .sortedByDescending { outputProbabilities[it] }
            .take(k)
            .mapNotNull { idxToLabelMap[it] }
            .filter { it != "unknown" }
    }

}
