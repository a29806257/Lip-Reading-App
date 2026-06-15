// 檔案路徑: app/src/main/java/com/example/lri/MainUI.kt

package com.example.lri

import android.content.Context
import android.content.Intent
import android.provider.Settings
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import com.example.lri.ui.theme.LRITheme

// =========================================================================
//  <<< 這裡是您所有的 UI 程式碼 >>>
// =========================================================================

/**
 * 檢查無障礙服務是否已啟用
 */
private fun isAccessibilityServiceEnabled(context: Context): Boolean {
    val prefString = Settings.Secure.getString(context.contentResolver, Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES)
    return prefString?.contains(context.packageName + "/" + MyAccessibilityService::class.java.name) ?: false
}

@Composable
fun MainLayout(
    isSwitchChecked: Boolean,
    onSwitchChanged: (Boolean) -> Unit
) {
    val context = LocalContext.current
    var isAccessibilityEnabled by remember {
        mutableStateOf(isAccessibilityServiceEnabled(context))
    }
    val lifecycleOwner = LocalLifecycleOwner.current

    // 監聽 Activity 生命週期，當 App 從背景切回前景時 (ON_RESUME)，
    // 重新檢查無障礙服務是否已被使用者開啟
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME) {
                isAccessibilityEnabled = isAccessibilityServiceEnabled(context)
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }

    Column(modifier = Modifier.fillMaxSize()) {
        // 使用說明元件
        Guid(modifier = Modifier.weight(1f))

        // 如果無障礙服務未啟用，顯示提示按鈕
        if (!isAccessibilityEnabled) {
            Column(
                modifier = Modifier.fillMaxWidth().padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text("請先開啟無障礙服務以啟用自動填入功能", color = Color.Red)
                Button(onClick = {
                    context.startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
                }) {
                    Text("前往設定")
                }
            }
        }

        // 懸浮視窗開關
        FloatingWindowSwitch(
            enabled = isAccessibilityEnabled, // 必須啟用無障礙才能操作開關
            isChecked = isSwitchChecked && isAccessibilityEnabled, // 兩者都啟用才顯示 "on"
            onCheckedChange = onSwitchChanged
        )
    }
}

@Composable
fun Guid(modifier: Modifier = Modifier) {
    Column(
        modifier = modifier.fillMaxWidth(),
    ) {
        Row(
            modifier = Modifier.weight(2f),
            verticalAlignment = Alignment.CenterVertically
        ){
            Image(
                modifier = Modifier
                    .fillMaxHeight()
                    .weight(3f),
                painter = painterResource(R.drawable.step1),
                contentDescription = "step1image"
            )
            Text(
                text = stringResource(R.string.step1),
                fontSize = 15.sp,
                modifier = Modifier.weight(7f)
            )
        }

        Row(
            modifier = Modifier.weight(0.5f).fillMaxWidth(),
        ) {
            Box(
                modifier = Modifier.weight(3f),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Filled.KeyboardArrowDown,
                    contentDescription = "ArrowDown",
                    modifier = Modifier.size(30.dp)
                )
            }
            Spacer(modifier = Modifier.weight(7f))
        }

        Row(
            modifier = Modifier.weight(4f),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Image(
                modifier = Modifier
                    .fillMaxHeight()
                    .weight(3f),
                painter = painterResource(R.drawable.step2),
                contentDescription = "step2image"
            )
            Column(modifier = Modifier.weight(7f)) {
                Text(text = stringResource(R.string.step2_1), fontSize = 15.sp)
                Spacer(modifier = Modifier.height(16.dp))
                Text(text = stringResource(R.string.step2_2), fontSize = 15.sp)
            }
        }

        Row(
            modifier = Modifier.weight(0.5f).fillMaxWidth(),
        ) {
            Box(
                modifier = Modifier.weight(3f),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Filled.KeyboardArrowDown,
                    contentDescription = "ArrowDown",
                    modifier = Modifier.size(30.dp)
                )
            }
            Spacer(modifier = Modifier.weight(7f))
        }

        Row(
            modifier = Modifier.weight(4f),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Image(
                modifier = Modifier
                    .fillMaxHeight()
                    .weight(3f),
                painter = painterResource(R.drawable.step3),
                contentDescription = "step3image"
            )
            Column(modifier = Modifier.weight(7f)) {
                Text(text = stringResource(R.string.step3_1), fontSize = 15.sp)
                Spacer(modifier = Modifier.height(16.dp))
                Text(text = stringResource(R.string.step3_2), fontSize = 15.sp)
                Spacer(modifier = Modifier.height(16.dp))
                Text(text = stringResource(R.string.step3_3), fontSize = 15.sp)
                Text(
                    text = stringResource(R.string.step3_3_2),
                    fontSize = 15.sp,
                    modifier = Modifier.padding(start = 15.dp)
                )
            }
        }

        Row(
            modifier = Modifier.weight(0.5f).fillMaxWidth(),
        ) {
            Box(
                modifier = Modifier.weight(3f),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Filled.KeyboardArrowDown,
                    contentDescription = "ArrowDown",
                    modifier = Modifier.size(30.dp)
                )
            }
            Spacer(modifier = Modifier.weight(7f))
        }

        Row(
            modifier = Modifier.weight(3f),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Image(
                modifier = Modifier
                    .fillMaxHeight()
                    .weight(3f),
                painter = painterResource(R.drawable.step4),
                contentDescription = "step4image"
            )
            Text(
                text = stringResource(R.string.step4),
                fontSize = 15.sp,
                modifier = Modifier.weight(7f)
            )
        }
    }
}


@Composable
fun FloatingWindowSwitch(
    enabled: Boolean,
    isChecked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(16.dp),
        horizontalArrangement = Arrangement.SpaceEvenly,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = stringResource(R.string.FloatingWindowSwitch),
            fontSize = 25.sp,
            fontWeight = FontWeight.Bold
        )
        Switch(
            enabled = enabled,
            checked = isChecked,
            onCheckedChange = onCheckedChange
        )
    }
}

@Preview(showBackground = true)
@Composable
fun MainLayoutPreview() {
    LRITheme {
        MainLayout(isSwitchChecked = false, onSwitchChanged = {})
    }
}