package com.speakerdetect

import android.Manifest
import android.animation.ObjectAnimator
import android.annotation.SuppressLint
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.pm.PackageManager
import android.graphics.Color
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.*
import android.view.Gravity
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import kotlinx.coroutines.*

class MainActivity : AppCompatActivity(), SpeakerEngine.Listener {

    companion object {
        private const val SAMPLE_RATE = 16000
        private const val CHANNEL = AudioFormat.CHANNEL_IN_MONO
        private const val ENCODING = AudioFormat.ENCODING_PCM_FLOAT
        private const val PROCESS_INTERVAL_MS = 1500L // Analyze every 1.5 seconds (faster!)
        private const val BUFFER_SECONDS = 3 // Process 3 seconds of audio (more responsive)
        private const val NOTIF_CHANNEL = "speaker_change"
        private val COLORS = intArrayOf(
            0xFFFF6B35.toInt(), 0xFF4ECDC4.toInt(), 0xFFFFE66D.toInt(),
            0xFFA78BFA.toInt(), 0xFFF38181.toInt(), 0xFF34D399.toInt(),
            0xFFFCBAD3.toInt(), 0xFF60A5FA.toInt()
        )
    }

    private lateinit var engine: SpeakerEngine
    private var audioRecord: AudioRecord? = null
    private var running = false
    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    // Audio ring buffer (3 seconds at 16kHz)
    private val audioBuffer = FloatArray(SAMPLE_RATE * BUFFER_SECONDS)
    private var bufferWritePos = 0
    private var bufferFilled = false

    // UI elements
    private lateinit var speakerLetter: TextView
    private lateinit var speakerName: TextView
    private lateinit var statusText: TextView
    private lateinit var debugText: TextView
    private lateinit var energyBar: View
    private lateinit var speakerBox: FrameLayout
    private lateinit var chipsBar: LinearLayout
    private lateinit var btnStart: Button
    private lateinit var flashOverlay: View

    private var lastSpeaker = -1
    private var analysisCount = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Find views
        speakerLetter = findViewById(R.id.speakerLetter)
        speakerName = findViewById(R.id.speakerName)
        statusText = findViewById(R.id.statusText)
        debugText = findViewById(R.id.debugText)
        energyBar = findViewById(R.id.energyBar)
        speakerBox = findViewById(R.id.speakerBox)
        chipsBar = findViewById(R.id.chipsBar)
        btnStart = findViewById(R.id.btnStart)
        flashOverlay = findViewById(R.id.flashOverlay)

        // Make debug text more visible
        debugText.textSize = 12f
        debugText.setTextColor(Color.parseColor("#888888"))

        // Setup notification channel
        createNotificationChannel()

        // Initialize engine
        engine = SpeakerEngine(this)
        engine.listener = this

        // Load model
        scope.launch {
            engine.loadModel()
        }

        // Button listeners
        btnStart.setOnClickListener { toggle() }
        findViewById<Button>(R.id.btnReset).setOnClickListener { resetAll() }
    }

    private fun toggle() {
        if (running) stopRecording() else startRecording()
    }

    @SuppressLint("MissingPermission")
    private fun startRecording() {
        if (!engine.modelLoaded) {
            statusText.text = "MODEL NOT READY — PLEASE WAIT"
            return
        }

        // Check permissions
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(
                Manifest.permission.RECORD_AUDIO,
                Manifest.permission.POST_NOTIFICATIONS
            ), 1)
            return
        }

        val minBufSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING)

        // Try UNPROCESSED source first (raw audio on Pixel)
        audioRecord = try {
            AudioRecord(
                MediaRecorder.AudioSource.UNPROCESSED,
                SAMPLE_RATE, CHANNEL, ENCODING,
                maxOf(minBufSize, SAMPLE_RATE * 2 * 4)
            ).also {
                if (it.state != AudioRecord.STATE_INITIALIZED) {
                    it.release()
                    throw Exception("UNPROCESSED not supported")
                }
            }
        } catch (e: Exception) {
            // Fallback to VOICE_RECOGNITION (less processing than MIC)
            try {
                AudioRecord(
                    MediaRecorder.AudioSource.VOICE_RECOGNITION,
                    SAMPLE_RATE, CHANNEL, ENCODING,
                    maxOf(minBufSize, SAMPLE_RATE * 2 * 4)
                )
            } catch (e2: Exception) {
                // Last resort: default MIC
                AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    SAMPLE_RATE, CHANNEL, ENCODING,
                    maxOf(minBufSize, SAMPLE_RATE * 2 * 4)
                )
            }
        }

        audioRecord?.startRecording()
        running = true
        bufferWritePos = 0
        bufferFilled = false
        analysisCount = 0

        btnStart.text = "STOP"
        btnStart.setTextColor(Color.parseColor("#FF4444"))
        btnStart.setBackgroundColor(Color.parseColor("#330000"))
        speakerLetter.text = "..."
        speakerLetter.setTextColor(Color.parseColor("#333333"))
        speakerName.text = "LISTENING"
        speakerName.setTextColor(Color.parseColor("#444444"))
        statusText.text = "RECORDING — FIRST ANALYSIS IN 2s"

        // Start audio capture thread
        scope.launch(Dispatchers.IO) {
            captureAudioLoop()
        }

        // Start periodic analysis
        scope.launch {
            while (running) {
                delay(PROCESS_INTERVAL_MS)
                if (!running) break
                analyzeCurrentAudio()
            }
        }
    }

    private fun stopRecording() {
        running = false
        try {
            audioRecord?.stop()
            audioRecord?.release()
        } catch (_: Exception) {}
        audioRecord = null

        btnStart.text = "START"
        btnStart.setTextColor(Color.parseColor("#00FF88"))
        btnStart.setBackgroundColor(Color.parseColor("#003300"))
        statusText.text = "STOPPED"
    }

    private fun captureAudioLoop() {
        val readBuffer = FloatArray(1024)
        while (running) {
            val read = audioRecord?.read(readBuffer, 0, readBuffer.size, AudioRecord.READ_BLOCKING) ?: 0
            if (read > 0) {
                synchronized(audioBuffer) {
                    for (i in 0 until read) {
                        audioBuffer[bufferWritePos] = readBuffer[i]
                        bufferWritePos = (bufferWritePos + 1) % audioBuffer.size
                        if (bufferWritePos == 0) bufferFilled = true
                    }
                }

                // Update energy bar on UI thread
                var rms = 0f
                for (i in 0 until read) rms += readBuffer[i] * readBuffer[i]
                rms = kotlin.math.sqrt(rms / read)
                val db = 20 * kotlin.math.log10(maxOf(rms, 1e-10f))
                val pct = maxOf(0f, minOf(1f, (db + 60) / 40f))

                runOnUiThread {
                    val params = energyBar.layoutParams
                    params.width = (speakerBox.width * pct).toInt().coerceAtLeast(1)
                    energyBar.layoutParams = params
                }
            }
        }
    }

    private suspend fun analyzeCurrentAudio() {
        // Get audio from ring buffer
        val audio: FloatArray
        synchronized(audioBuffer) {
            val totalSamples = if (bufferFilled) audioBuffer.size else bufferWritePos
            if (totalSamples < SAMPLE_RATE) return // Need at least 1 second
            audio = FloatArray(totalSamples)
            if (bufferFilled) {
                val firstPart = audioBuffer.size - bufferWritePos
                System.arraycopy(audioBuffer, bufferWritePos, audio, 0, firstPart)
                System.arraycopy(audioBuffer, 0, audio, firstPart, bufferWritePos)
            } else {
                System.arraycopy(audioBuffer, 0, audio, 0, totalSamples)
            }
        }

        analysisCount++
        engine.processAudio(audio)
    }

    // === SpeakerEngine.Listener ===

    override fun onModelLoading(progress: String) {
        runOnUiThread {
            statusText.text = progress
            debugText.text = "Loading model..."
        }
    }

    override fun onModelReady() {
        runOnUiThread {
            speakerName.text = "TAP START"
            statusText.text = "MODEL READY — WESPEAKER 256-DIM"
            debugText.text = "ONNX loaded • Tap START to begin detecting speakers"
            btnStart.text = "START"
            btnStart.isEnabled = true
            findViewById<TextView>(R.id.modelBadge).setTextColor(Color.parseColor("#00FF88"))
        }
    }

    override fun onModelError(error: String) {
        runOnUiThread {
            statusText.text = "ERROR: $error"
            debugText.text = "Model failed to load. Check internet connection and tap RESET."
            speakerName.text = "MODEL FAILED"
            speakerName.setTextColor(Color.parseColor("#FF4444"))
            // Allow retry via reset
            btnStart.isEnabled = false
        }
    }

    override fun onSpeakerDetected(speakerIndex: Int, isChange: Boolean, similarity: Float, debug: String) {
        runOnUiThread {
            val color = COLORS[speakerIndex % COLORS.size]
            val letter = ('A' + speakerIndex).toString()

            speakerLetter.text = letter
            speakerLetter.setTextColor(color)
            speakerName.text = "SPEAKER $letter"
            speakerName.setTextColor(color)
            statusText.text = "SPEAKER $letter · ${"%.0f".format(similarity * 100)}% match · #$analysisCount"
            debugText.text = debug

            // Update energy bar color
            energyBar.setBackgroundColor(color)

            if (isChange && lastSpeaker != -1) {
                // Speaker CHANGED — flash + vibrate + notify
                val fromLetter = ('A' + lastSpeaker).toString()
                doubleFlash()
                vibrate()
                sendNotification(fromLetter, letter)
            }
            lastSpeaker = speakerIndex

            // Update chips
            updateChips(engine.totalSpeakers, speakerIndex)
        }
    }

    override fun onNoSpeech(debug: String) {
        runOnUiThread {
            if (lastSpeaker >= 0) {
                statusText.text = "ACTIVE — NO VOICE DETECTED · #$analysisCount"
            } else {
                statusText.text = "ACTIVE — SPEAK NOW · #$analysisCount"
            }
            debugText.text = debug
        }
    }

    // === UI helpers ===

    private fun doubleFlash() {
        doFlash()
        flashOverlay.postDelayed({ doFlash() }, 250)
    }

    private fun doFlash() {
        flashOverlay.alpha = 0.8f
        ObjectAnimator.ofFloat(flashOverlay, "alpha", 0.8f, 0f).apply {
            duration = 200
            start()
        }
    }

    private fun vibrate() {
        val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            (getSystemService(VIBRATOR_MANAGER_SERVICE) as VibratorManager).defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            getSystemService(VIBRATOR_SERVICE) as Vibrator
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createWaveform(longArrayOf(0, 150, 100, 150), -1))
        }
    }

    private fun updateChips(total: Int, active: Int) {
        chipsBar.removeAllViews()
        for (i in 0 until total) {
            val chip = TextView(this).apply {
                text = ('A' + i).toString()
                textSize = 18f
                setTextColor(COLORS[i % COLORS.size])
                gravity = Gravity.CENTER
                val size = (44 * resources.displayMetrics.density).toInt()
                layoutParams = LinearLayout.LayoutParams(size, size).apply {
                    marginEnd = (4 * resources.displayMetrics.density).toInt()
                }
                background = android.graphics.drawable.GradientDrawable().apply {
                    cornerRadius = 12 * resources.displayMetrics.density
                    setStroke((2 * resources.displayMetrics.density).toInt(),
                        if (i == active) COLORS[i % COLORS.size] else Color.parseColor("#222222"))
                    setColor(if (i == active) Color.argb(32,
                        Color.red(COLORS[i % COLORS.size]),
                        Color.green(COLORS[i % COLORS.size]),
                        Color.blue(COLORS[i % COLORS.size])) else Color.parseColor("#0A0A0A"))
                }
                alpha = if (i == active) 1f else 0.3f
            }
            chipsBar.addView(chip)
        }
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(NOTIF_CHANNEL, "Speaker Changes",
            NotificationManager.IMPORTANCE_HIGH).apply {
            description = "Notifies when a different person starts speaking"
            enableVibration(true)
        }
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
    }

    private fun sendNotification(fromLetter: String, toLetter: String) {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
            != PackageManager.PERMISSION_GRANTED) return
        val notification = NotificationCompat.Builder(this, NOTIF_CHANNEL)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentTitle("Speaker Changed: $fromLetter → $toLetter")
            .setContentText("Speaker $toLetter is now talking")
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true)
            .build()
        NotificationManagerCompat.from(this).notify(System.currentTimeMillis().toInt(), notification)
    }

    private fun resetAll() {
        engine.reset()
        lastSpeaker = -1
        analysisCount = 0
        speakerLetter.text = "?"
        speakerLetter.setTextColor(Color.parseColor("#333333"))
        speakerName.text = if (engine.modelLoaded) "TAP START" else "LOADING..."
        speakerName.setTextColor(Color.parseColor("#444444"))
        statusText.text = if (engine.modelLoaded) "READY" else "RELOADING MODEL..."
        debugText.text = ""
        chipsBar.removeAllViews()

        // If model failed, try reloading
        if (!engine.modelLoaded) {
            btnStart.isEnabled = false
            scope.launch { engine.loadModel() }
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, results: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, results)
        if (requestCode == 1 && results.isNotEmpty() && results[0] == PackageManager.PERMISSION_GRANTED) {
            startRecording()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        running = false
        try { audioRecord?.release() } catch (_: Exception) {}
        engine.release()
        scope.cancel()
    }
}
