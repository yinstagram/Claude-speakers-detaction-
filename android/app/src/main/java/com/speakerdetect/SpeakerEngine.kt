package com.speakerdetect

import ai.onnxruntime.*
import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.net.URL
import kotlin.math.sqrt

/**
 * Speaker embedding engine using WeSpeaker ResNet34-LM via ONNX Runtime.
 * Uses NNAPI delegate on Pixel devices to leverage the Tensor TPU.
 *
 * Model: wespeaker-voxceleb-resnet34-LM (~26MB)
 * - Trained on VoxCeleb (7000+ speakers)
 * - Input: [1, numFrames, 80] fbank features
 * - Output: [1, 256] speaker embedding
 */
class SpeakerEngine(private val context: Context) {

    private var session: OrtSession? = null
    private val env = OrtEnvironment.getEnvironment()
    private val fbankExtractor = FbankExtractor()
    private val profiles = mutableListOf<SpeakerProfile>()
    var currentSpeaker: Int = -1
        private set

    data class SpeakerProfile(
        val embedding: FloatArray,
        var frozen: Boolean = false,
        val embeddings: MutableList<FloatArray> = mutableListOf()
    )

    companion object {
        private const val MODEL_URL =
            "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/main/voxceleb_resnet34_LM.onnx"
        private const val MODEL_FILE = "wespeaker_resnet34.onnx"
        private const val EMBEDDING_DIM = 256
        private const val SIMILARITY_THRESHOLD = 0.65f
        private const val FREEZE_AFTER = 10 // Freeze profile after 10 updates
    }

    interface Listener {
        fun onModelLoading(progress: String)
        fun onModelReady()
        fun onModelError(error: String)
        fun onSpeakerDetected(speakerIndex: Int, isChange: Boolean, similarity: Float, debug: String)
        fun onNoSpeech(debug: String)
    }

    var listener: Listener? = null

    /**
     * Download and load the ONNX model. Uses NNAPI for hardware acceleration.
     */
    suspend fun loadModel() = withContext(Dispatchers.IO) {
        try {
            val modelFile = File(context.filesDir, MODEL_FILE)

            if (!modelFile.exists()) {
                listener?.onModelLoading("Downloading AI model (26MB)...")
                val connection = URL(MODEL_URL).openConnection()
                connection.connectTimeout = 30000
                connection.readTimeout = 60000
                val totalSize = connection.contentLength
                var downloaded = 0L

                connection.getInputStream().use { input ->
                    modelFile.outputStream().use { output ->
                        val buffer = ByteArray(8192)
                        var bytesRead: Int
                        while (input.read(buffer).also { bytesRead = it } != -1) {
                            output.write(buffer, 0, bytesRead)
                            downloaded += bytesRead
                            if (totalSize > 0) {
                                val pct = (downloaded * 100 / totalSize).toInt()
                                listener?.onModelLoading("Downloading: $pct%")
                            }
                        }
                    }
                }
            }

            listener?.onModelLoading("Loading model into NNAPI...")

            // Configure ONNX Runtime with NNAPI (uses Tensor TPU on Pixel)
            val sessionOptions = OrtSession.SessionOptions().apply {
                try {
                    addNnapi() // Hardware acceleration on Android (Tensor TPU, GPU, DSP)
                } catch (e: Exception) {
                    // Fallback to CPU if NNAPI not available
                }
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }

            session = env.createSession(modelFile.absolutePath, sessionOptions)
            listener?.onModelReady()

        } catch (e: Exception) {
            listener?.onModelError(e.message ?: "Unknown error")
        }
    }

    /**
     * Process raw 16kHz mono PCM audio and detect speaker.
     */
    suspend fun processAudio(audio: FloatArray) = withContext(Dispatchers.Default) {
        val sess = session ?: return@withContext

        // Extract fbank features
        val features = fbankExtractor.extract(audio)
        val numFrames = fbankExtractor.getNumFrames(audio.size)
        if (numFrames < 10) {
            listener?.onNoSpeech("Too short: $numFrames frames")
            return@withContext
        }

        // Check if there's actual voice (RMS energy check)
        var rms = 0f
        for (s in audio) rms += s * s
        rms = sqrt(rms / audio.size)
        if (rms < 0.005f) {
            listener?.onNoSpeech("dB:${(20 * kotlin.math.log10(rms.coerceAtLeast(1e-10f))).toInt()} (silence)")
            return@withContext
        }

        // Run ONNX model — input shape: [1, numFrames, 80]
        val inputTensor = OnnxTensor.createTensor(
            env,
            arrayOf(Array(numFrames) { f ->
                FloatArray(80) { b -> features[f * 80 + b] }
            })
        )

        val results = sess.run(mapOf("feats" to inputTensor))
        val outputTensor = results[0] as OnnxTensor

        // Extract 256-dim embedding
        @Suppress("UNCHECKED_CAST")
        val rawOutput = outputTensor.value
        val embedding: FloatArray = when (rawOutput) {
            is Array<*> -> {
                @Suppress("UNCHECKED_CAST")
                val arr = rawOutput as Array<FloatArray>
                arr[0]
            }
            else -> FloatArray(EMBEDDING_DIM)
        }

        // L2 normalize the embedding
        val norm = sqrt(embedding.sumOf { (it * it).toDouble() }.toFloat())
        if (norm > 0) for (i in embedding.indices) embedding[i] /= norm

        inputTensor.close()
        results.close()

        // Compare against profiles
        matchSpeaker(embedding, rms)
    }

    private fun matchSpeaker(embedding: FloatArray, rms: Float) {
        val dbStr = "dB:${(20 * kotlin.math.log10(rms.coerceAtLeast(1e-10f))).toInt()}"

        if (profiles.isEmpty()) {
            // First speaker
            profiles.add(SpeakerProfile(embedding.clone()).also {
                it.embeddings.add(embedding.clone())
            })
            currentSpeaker = 0
            listener?.onSpeakerDetected(0, true, 1f, "$dbStr profiles:1")
            return
        }

        // Find best matching profile
        var bestIdx = -1
        var bestSim = -1f
        val simTexts = mutableListOf<String>()

        for (i in profiles.indices) {
            val sim = cosineSimilarity(embedding, profiles[i].embedding)
            simTexts.add("${('A' + i)}:${"%.3f".format(sim)}")
            if (sim > bestSim) {
                bestSim = sim
                bestIdx = i
            }
        }

        val debug = "$dbStr thr:$SIMILARITY_THRESHOLD ${simTexts.joinToString(" ")}"

        if (bestSim >= SIMILARITY_THRESHOLD) {
            // Matches existing speaker
            val isChange = bestIdx != currentSpeaker
            currentSpeaker = bestIdx

            // Update profile if not frozen
            val p = profiles[bestIdx]
            if (!p.frozen) {
                p.embeddings.add(embedding.clone())
                if (p.embeddings.size >= FREEZE_AFTER) {
                    // Compute final embedding as average of all collected
                    val avg = FloatArray(EMBEDDING_DIM)
                    for (e in p.embeddings) for (i in avg.indices) avg[i] += e[i]
                    for (i in avg.indices) avg[i] /= p.embeddings.size
                    // Re-normalize
                    val n = sqrt(avg.sumOf { (it * it).toDouble() }.toFloat())
                    if (n > 0) for (i in avg.indices) avg[i] /= n
                    p.embeddings.clear()
                    profiles[bestIdx] = SpeakerProfile(avg, frozen = true)
                }
            }

            listener?.onSpeakerDetected(bestIdx, isChange, bestSim, debug)
        } else {
            // New speaker
            profiles.add(SpeakerProfile(embedding.clone()).also {
                it.embeddings.add(embedding.clone())
            })
            currentSpeaker = profiles.size - 1
            listener?.onSpeakerDetected(currentSpeaker, true, bestSim, "$debug NEW")
        }
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        var dot = 0f; var na = 0f; var nb = 0f
        for (i in a.indices) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (sqrt(na) * sqrt(nb) + 1e-10f)
    }

    fun reset() {
        profiles.clear()
        currentSpeaker = -1
    }

    fun release() {
        session?.close()
        session = null
    }

    val totalSpeakers: Int get() = profiles.size
}
