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
        private const val PROCESS_INTERVAL_MS = 2500L // Analyze every 2.5 seconds
        private const val BUFFER_SECONDS = 5 // Process 5 seconds of audio
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

    // Audio ring buffer (5 seconds at 16kHz)
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
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.UNPROCESSED, // Raw audio — no processing!
            SAMPLE_RATE, CHANNEL, ENCODING,
            maxOf(minBufSize, SAMPLE_RATE * 2 * 4) // At least 2 seconds buffer
        ).also {
            if (it.state != AudioRecord.STATE_INITIALIZED) {
                // Fallback to default source if UNPROCESSED not supported
                audioRecord = AudioRecord(
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

        btnStart.text = "STOP"
        btnStart.setTextColor(Color.parseColor("#FF4444"))
        btnStart.setBackgroundColor(Color.parseColor("#330000"))
        speakerLetter.text = "..."
        speakerLetter.setTextColor(Color.parseColor("#333333"))
        speakerName.text = "LISTENING"
        speakerName.setTextColor(Color.parseColor("#444444"))
        statusText.text = "RECORDING — FIRST ANALYSIS IN 3s"

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
        audioRecord?.stop()
        audioRecord?.release()
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
                // Copy from writePos to end, then start to writePos
                val firstPart = audioBuffer.size - bufferWritePos
                System.arraycopy(audioBuffer, bufferWritePos, audio, 0, firstPart)
                System.arraycopy(audioBuffer, 0, audio, firstPart, bufferWritePos)
            } else {
                System.arraycopy(audioBuffer, 0, audio, 0, totalSamples)
            }
        }

        engine.processAudio(audio)
    }

    // === SpeakerEngine.Listener ===

    override fun onModelLoading(progress: String) {
        runOnUiThread {
            statusText.text = progress
        }
    }

    override fun onModelReady() {
        runOnUiThread {
            speakerName.text = "TAP START"
            statusText.text = "ONNX NNAPI · WESPEAKER · 256-DIM"
            btnStart.text = "START"
            btnStart.isEnabled = true
            findViewById<TextView>(R.id.modelBadge).setTextColor(Color.parseColor("#00FF88"))
        }
    }

    override fun onModelError(error: String) {
        runOnUiThread {
            statusText.text = "ERROR: $error"
            speakerName.text = "MODEL FAILED"
            speakerName.setTextColor(Color.parseColor("#FF4444"))
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
            statusText.text = "SPEAKER $letter · ${"%.1f".format(similarity * 100)}%"
            debugText.text = debug
            debugText.setTextColor(Color.argb(60, Color.red(color), Color.green(color), Color.blue(color)))

            // Update energy bar color
            energyBar.setBackgroundColor(color)

            if (isChange && lastSpeaker != -1) {
                // Speaker CHANGED — flash + vibrate + notify
                doubleFlash()
                vibrate()
                sendNotification(letter)
            }
            lastSpeaker = speakerIndex

            // Update chips
            updateChips(engine.totalSpeakers, speakerIndex)
        }
    }

    override fun onNoSpeech(debug: String) {
        runOnUiThread {
            statusText.text = if (lastSpeaker >= 0) "ACTIVE — WAITING" else "ACTIVE — SPEAK NOW"
            debugText.text = debug
        }
    }

    // === UI helpers ===

    private fun doubleFlash() {
        doFlash()
        flashOverlay.postDelayed({ doFlash() }, 250)
    }

    private fun doFlash() {
        flashOverlay.alpha = 0.7f
        ObjectAnimator.ofFloat(flashOverlay, "alpha", 0.7f, 0f).apply {
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
            vibrator.vibrate(VibrationEffect.createWaveform(longArrayOf(0, 100, 80, 100), -1))
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

    private fun sendNotification(letter: String) {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
            != PackageManager.PERMISSION_GRANTED) return
        val notification = NotificationCompat.Builder(this, NOTIF_CHANNEL)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentTitle("Speaker Changed")
            .setContentText("Now speaking: Speaker $letter")
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true)
            .build()
        NotificationManagerCompat.from(this).notify(1, notification)
    }

    private fun resetAll() {
        engine.reset()
        lastSpeaker = -1
        speakerLetter.text = "?"
        speakerLetter.setTextColor(Color.parseColor("#333333"))
        speakerName.text = "TAP START"
        speakerName.setTextColor(Color.parseColor("#444444"))
        statusText.text = "READY"
        chipsBar.removeAllViews()
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
        audioRecord?.release()
        engine.release()
        scope.cancel()
    }
}
