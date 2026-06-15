package com.example.lri

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class LipReadingManager(private val context: Context) {

    private val isMockMode = false

    private var tfliteInterpreter: Interpreter? = null
    private val lipProcessor = LipReadingProcessor(context)
    private val smartConverter = SmartPinyinConverter(context)

    private val INPUT_SIZE = 88
    private val NUM_FRAMES = 40
    private val NUM_CLASSES = 187
    private val MODEL_NAME = "final_model_float32.tflite"

    // 加入狀態標記，防止模型還沒載入完就被呼叫
    var isModelReady = false
        private set

    init {
        if (!isMockMode) {
            // 解決 Input Latency：將載入模型的沉重 I/O 操作移至背景執行
            CoroutineScope(Dispatchers.IO).launch {
                loadModel()
            }
        }
    }

    private fun loadModel() {
        try {
            val assetFileDescriptor = context.assets.openFd(MODEL_NAME)
            val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = fileInputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength
            val mappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }
            tfliteInterpreter = Interpreter(mappedByteBuffer, options)
            isModelReady = true
            Log.d("LRI_DEBUG", "✅ TFLite 模型載入成功 (背景)")
        } catch (e: Exception) {
            Log.e("LRI_DEBUG", "❌ TFLite 模型載入失敗", e)
        }
    }

    // 解決 Lifecycle Stability：加入安全釋放資源的方法
    fun close() {
        try {
            tfliteInterpreter?.close()
            tfliteInterpreter = null
            isModelReady = false
            Log.d("LRI_DEBUG", "✅ TFLite 模型已安全釋放")
        } catch (e: Exception) {
            Log.e("LRI_DEBUG", "釋放模型時發生錯誤", e)
        }
    }

    fun recognizeSentence(wordBitmaps: List<List<Bitmap>>): List<String> {
        if (!isModelReady) return listOf("系統準備中，請稍後...")

        val pinyinSequenceCandidates = ArrayList<List<String>>()

        for ((index, frames) in wordBitmaps.withIndex()) {
            val inputBuffer = preprocessBitmaps(frames)

            if (inputBuffer != null) {
                var topPinyins = listOf("unknown")

                if (!isMockMode && tfliteInterpreter != null) {
                    try {
                        val outputArray = Array(1) { FloatArray(NUM_CLASSES) }
                        tfliteInterpreter?.run(inputBuffer, outputArray)

                        topPinyins = lipProcessor.getTopKPinyins(outputArray[0], 3)
                        lipProcessor.decodeAndLog(outputArray[0], index + 1)
                    } catch (e: Exception) {
                        Log.e("LRI_DEBUG", "推論崩潰", e)
                    }
                }
                if (topPinyins.isNotEmpty() && topPinyins[0] != "unknown") {
                    pinyinSequenceCandidates.add(topPinyins)
                }
            }
        }
        if (pinyinSequenceCandidates.isEmpty()) return listOf("無法辨識拼音")
        return smartConverter.convertToSentence(pinyinSequenceCandidates)
    }

    private fun preprocessBitmaps(frames: List<Bitmap>): ByteBuffer? {
        val bufferSize = 1 * NUM_FRAMES * INPUT_SIZE * INPUT_SIZE * 1 * 4
        val byteBuffer = ByteBuffer.allocateDirect(bufferSize)
        byteBuffer.order(ByteOrder.nativeOrder())

        // 保持你原本正確的記憶體排列方式
        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                for (i in 0 until NUM_FRAMES) {
                    var r = 0
                    if (i < frames.size) {
                        val bitmap = frames[i]
                        if (x < bitmap.width && y < bitmap.height) {
                            val pixel = bitmap.getPixel(x, y)
                            r = (pixel shr 16) and 0xFF
                        }
                    }
                    // TODO: 這裡請換回你確認過的正規化公式 (除以 255 或是 -0.5 / 0.5 等)
                    byteBuffer.putFloat(r / 255.0f)
                }
            }
        }
        byteBuffer.rewind()
        return byteBuffer
    }
}