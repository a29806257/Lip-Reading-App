// 檔案路徑: app/src/main/java/com/example/lri/MainActivity.kt

package com.example.lri

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import androidx.core.net.toUri
import com.example.lri.ui.theme.LRITheme

class MainActivity : ComponentActivity() {

    // =========================================================================
    //  <<< 這裡是您所有的 邏輯 程式碼 >>>
    // =========================================================================

    // 儲存懸浮視窗開關的狀態
    private val uiState = mutableStateOf(false)
    private var isWindowShown: Boolean
        get() = uiState.value
        set(value) {
            uiState.value = value
        }

    // 定義 App 需要的權限 (已移除 RECORD_AUDIO)
    private val requiredPermissions = mutableListOf(
        Manifest.permission.CAMERA
    ).apply {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            add(Manifest.permission.POST_NOTIFICATIONS)
        }
    }.toTypedArray()

    // 註冊一個權限請求的啟動器
    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
            if (permissions.all { it.value }) {
                // 如果所有權限都同意了，啟動服務
                startFloatingService()
            } else {
                // 如果有任何權限被拒絕
                Toast.makeText(this, "需要所有權限才能啟用此功能", Toast.LENGTH_LONG).show()
                isWindowShown = false
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // setContent 是 Activity (邏輯) 和 Composable (UI) 之間的橋樑
        setContent {
            LRITheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    // 呼叫 MainUI.kt 中定義的 MainLayout
                    MainLayout(
                        isSwitchChecked = isWindowShown,
                        onSwitchChanged = { isChecked ->
                            // 這是從 UI 傳回來的點擊事件
                            if (isChecked) {
                                checkPermissionsAndStartService()
                            } else {
                                stopFloatingService()
                            }
                        }
                    )
                }
            }
        }
    }

    /**
     * 檢查所有權限並啟動服務
     */
    private fun checkPermissionsAndStartService() {
        // 1. 檢查懸浮視窗權限
        if (!Settings.canDrawOverlays(this)) {
            Toast.makeText(this, "請先授予懸浮視窗權限", Toast.LENGTH_SHORT).show()
            val intent = Intent(
                Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                "package:$packageName".toUri()
            )
            startActivity(intent)
            isWindowShown = false // 將開關彈回 "off"
            return
        }

        // 2. 檢查相機和通知權限
        val allPermissionsGranted = requiredPermissions.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }

        if (allPermissionsGranted) {
            // 如果權限都已授予，直接啟動服務
            startFloatingService()
        } else {
            // 否則，跳出權限請求視窗
            requestPermissionLauncher.launch(requiredPermissions)
        }
    }

    /**
     * 啟動懸浮視窗服務
     */
    private fun startFloatingService() {
        startService(Intent(this, FloatingWindowService::class.java))
        isWindowShown = true
        Toast.makeText(this, "懸浮視窗已啟動", Toast.LENGTH_SHORT).show()
    }

    /**
     * 停止懸浮視窗服務
     */
    private fun stopFloatingService() {
        stopService(Intent(this, FloatingWindowService::class.java))
        isWindowShown = false
    }
}