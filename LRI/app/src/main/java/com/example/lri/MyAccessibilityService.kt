package com.example.lri

import android.accessibilityservice.AccessibilityService
import android.content.Intent
import android.os.Bundle
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo // 確保此匯入存在
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.cancel

class MyAccessibilityService : AccessibilityService() {

    // 服務的 CoroutineScope
    private val serviceScope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    companion object {
        // 保留緩衝區 (Buffer)，這是確保 Flow 正常運作的關鍵
        val pasteTextFlow = MutableSharedFlow<String>(extraBufferCapacity = 5)
    }

    override fun onServiceConnected() {
        super.onServiceConnected()

        // 服務連接時，開始監聽 Flow
        serviceScope.launch {
            pasteTextFlow.collect { textToPaste ->
                performPaste(textToPaste)
            }
        }
    }

    /**
     * 執行貼上文字的操作
     */
    private fun performPaste(text: String) {
        // 取得當前活動視窗的根節點，如果失敗則提早返回
        val rootNode = rootInActiveWindow ?: return

        // 策略 1: 優先尋找「輸入焦點」
        var targetNode = findFocus(AccessibilityNodeInfo.FOCUS_INPUT)

        // 策略 2: 如果找不到「輸入焦點」，則備用尋找「無障礙焦點」
        if (targetNode == null) {
            targetNode = rootNode.findFocus(AccessibilityNodeInfo.FOCUS_ACCESSIBILITY)
        }

        // 如果找到了目標節點
        targetNode?.let { node ->
            // 必須檢查節點是否可編輯
            if (node.isEditable) {
                val arguments = Bundle()
                arguments.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text)
                node.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, arguments)
            }
        }
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) { }

    override fun onInterrupt() { }

    override fun onUnbind(intent: Intent?): Boolean {
        // 服務被關閉或解除綁定時，取消所有 Coroutine，防止記憶體洩漏
        serviceScope.cancel()
        return super.onUnbind(intent)
    }
}