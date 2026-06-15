package com.example.lri

import android.content.Context
import android.util.JsonReader
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.InputStreamReader
import kotlin.math.log10

class SmartPinyinConverter(context: Context) {

    // 拼音 -> 詞列表 (例如 "zhong1jie4" -> ["中介"])
    private val pinyinToWords = mutableMapOf<String, List<String>>()
    // 詞 -> 分數
    private val wordFreq = mutableMapOf<String, Double>()

    init {
        // 1. 使用 Coroutine 將讀取動作移至背景執行緒 (IO)，絕不卡死 UI 導致掉幀
        CoroutineScope(Dispatchers.IO).launch {
            loadDataOptimized(context)
        }
    }

    private fun loadDataOptimized(context: Context) {
        try {
            val inputStream = context.assets.open("smart_dict.json")
            // 2. 使用 JsonReader 進行串流 (Streaming) 解析，大幅降低記憶體消耗與 GC 頻率
            val reader = JsonReader(InputStreamReader(inputStream, "UTF-8"))

            reader.beginObject()
            while (reader.hasNext()) {
                val rootKey = reader.nextName()
                when (rootKey) {
                    "mapping" -> parseMapping(reader)
                    "freq" -> parseFreq(reader)
                    else -> reader.skipValue()
                }
            }
            reader.endObject()
            reader.close()

            Log.d("LRI", "智慧字典載入完成！拼音組合: ${pinyinToWords.size}, 詞頻數量: ${wordFreq.size}")
        } catch (e: Exception) {
            Log.e("LRI", "讀取 smart_dict.json 失敗", e)
        }
    }

    private fun parseMapping(reader: JsonReader) {
        reader.beginObject()
        while (reader.hasNext()) {
            val pinyinKey = reader.nextName()
            val wordsList = mutableListOf<String>()

            reader.beginArray()
            while (reader.hasNext()) {
                wordsList.add(reader.nextString())
            }
            reader.endArray()

            pinyinToWords[pinyinKey] = wordsList
        }
        reader.endObject()
    }

    private fun parseFreq(reader: JsonReader) {
        reader.beginObject()
        while (reader.hasNext()) {
            val word = reader.nextName()
            val freq = reader.nextDouble()
            wordFreq[word] = freq
        }
        reader.endObject()
    }

    // 內部資料類別，用來儲存路徑與分數

    private data class PathNode(val score: Double, val words: List<String>)

    // 用來儲存交叉組合後的拼音字串與懲罰分數
    private data class PinyinCombo(val pinyinStr: String, val penalty: Double)

    /**
     * Beam Search 最佳路徑演算法 (支援多個候選選項)
     * 將拼音序列轉換成最可能的前 topK 個中文句子
     */
    fun convertToSentence(pinyinCandidatesList: List<List<String>>, topK: Int = 3): List<String> {
        if (pinyinCandidatesList.isEmpty()) return emptyList()

        val n = pinyinCandidatesList.size
        val dp = Array(n + 1) { mutableListOf<PathNode>() }
        dp[0].add(PathNode(0.0, emptyList()))

        for (i in 0 until n) {
            if (dp[i].isEmpty()) continue

            dp[i].sortByDescending { it.score }
            if (dp[i].size > topK) {
                val bestPaths = dp[i].take(topK)
                dp[i].clear()
                dp[i].addAll(bestPaths)
            }

            var madeJump = false

            for (len in 1..6) {
                if (i + len > n) break

                // 將未來 len 個字的 Top 3 拼音進行交叉組合
                val combos = generateCombinations(pinyinCandidatesList.subList(i, i + len))

                for (combo in combos) {
                    val candidates = pinyinToWords[combo.pinyinStr]

                    if (candidates != null && candidates.isNotEmpty()) {
                        madeJump = true
                        val topCandidates = candidates.sortedByDescending { wordFreq[it] ?: 0.0 }.take(topK)

                        for (word in topCandidates) {
                            val scoreVal = wordFreq[word]?.takeIf { it > 0 } ?: 0.0001

                            // 總分 = 字典詞頻分數 + 拼音排名的懲罰分數
                            val stepScore = log10(scoreVal) + combo.penalty

                            for (path in dp[i]) {
                                val newScore = path.score + stepScore
                                val newWords = path.words + listOf(word)
                                dp[i + len].add(PathNode(newScore, newWords))
                            }
                        }
                    }
                }
            }

            if (!madeJump) {
                val penaltyScore = -10.0
                for (path in dp[i]) {
                    // 如果所有組合都查不到字典，強制取模型第 1 名的拼音盲接
                    dp[i + 1].add(PathNode(path.score + penaltyScore, path.words + listOf(pinyinCandidatesList[i][0])))
                }
            }
        }

        val finalPaths = dp[n]
        if (finalPaths.isEmpty()) return listOf("無法組合字詞")

        finalPaths.sortByDescending { it.score }
        val results = finalPaths.map { it.words.joinToString("") }.distinct().take(topK)

        return if (results.isNotEmpty()) results else listOf("無法組合字詞")
    }

    /**
     * 遞迴產生所有可能的拼音交叉組合，並計算懲罰分數
     */
    private fun generateCombinations(candidatesList: List<List<String>>): List<PinyinCombo> {
        var currentCombos = listOf(PinyinCombo("", 0.0))

        for (candidates in candidatesList) {
            val nextCombos = mutableListOf<PinyinCombo>()
            for (prefix in currentCombos) {
                for ((index, pinyin) in candidates.withIndex()) {
                    // 第 1 名 (index 0) 扣 0 分
                    // 第 2 名 (index 1) 扣 2.0 分
                    // 第 3 名 (index 2) 扣 4.0 分
                    // 這能確保模型有信心時聽模型的，模型猶豫時聽字典的
                    val penalty = prefix.penalty - (index * 5.0)
                    nextCombos.add(PinyinCombo(prefix.pinyinStr + pinyin, penalty))
                }
            }
            currentCombos = nextCombos
        }
        return currentCombos
    }
}