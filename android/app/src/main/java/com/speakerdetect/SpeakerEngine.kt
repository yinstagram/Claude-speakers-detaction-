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
    private var inputName = "feats"
    var modelLoaded = false
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
        private const val SIMILARITY_THRESHOLD = 0.55f
        private const val FREEZE_AFTER = 8
        private const val MIN_MODEL_SIZE = 20_000_000L // 20MB minimum for valid model
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
     * Download and load the ONNX model. Tries NNAPI first, falls back to CPU.
     */
    suspend fun loadModel() = withContext(Dispatchers.IO) {
        try {
            val modelFile = File(context.filesDir, MODEL_FILE)

            // Check if existing model file is valid (not corrupt/truncated)
            if (modelFile.exists() && modelFile.length() < MIN_MODEL_SIZE) {
                listener?.onModelLoading("Removing corrupt model file...")
                modelFile.delete()
            }

            if (!modelFile.exists()) {
                listener?.onModelLoading("Downloading AI model (26MB)...")
                downloadModel(modelFile)
            }

            // Verify download
            if (!modelFile.exists() || modelFile.length() < MIN_MODEL_SIZE) {
                modelFile.delete()
                listener?.onModelError("Download failed — file too small (${modelFile.length()} bytes). Tap RESET and try again.")
                return@withContext
            }

            // Try NNAPI first (hardware acceleration on Pixel Tensor)
            var loaded = false
            try {
                listener?.onModelLoading("Loading model with NNAPI...")
                val nnapiOptions = OrtSession.SessionOptions().apply {
                    addNnapi()
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                }
                session = env.createSession(modelFile.absolutePath, nnapiOptions)
                loaded = true
                listener?.onModelLoading("NNAPI loaded!")
            } catch (e: Exception) {
                listener?.onModelLoading("NNAPI failed, using CPU: ${e.message?.take(50)}")
            }

            // Fallback to CPU if NNAPI failed
            if (!loaded) {
                try {
                    val cpuOptions = OrtSession.SessionOptions().apply {
                        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                    }
                    session = env.createSession(modelFile.absolutePath, cpuOptions)
                    loaded = true
                } catch (e: Exception) {
                    // Model file might be corrupt — delete and report
                    modelFile.delete()
                    listener?.onModelError("Model load failed: ${e.message?.take(80)}. Deleted file — tap RESET to retry.")
                    return@withContext
                }
            }

            // Auto-detect input tensor name from model metadata
            session?.let { sess ->
                val inputInfo = sess.inputInfo
                if (inputInfo.isNotEmpty()) {
                    inputName = inputInfo.keys.first()
                }
            }

            modelLoaded = true
            listener?.onModelReady()

        } catch (e: Exception) {
            listener?.onModelError("Unexpected: ${e.message?.take(100)}")
        }
    }

    private fun downloadModel(modelFile: File) {
        val tmpFile = File(context.filesDir, "$MODEL_FILE.tmp")
        try {
            val connection = URL(MODEL_URL).openConnection()
            connection.connectTimeout = 30000
            connection.readTimeout = 120000
            val totalSize = connection.contentLength

            connection.getInputStream().use { input ->
                tmpFile.outputStream().use { output ->
                    val buffer = ByteArray(16384)
                    var downloaded = 0L
                    var bytesRead: Int
                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        output.write(buffer, 0, bytesRead)
                        downloaded += bytesRead
                        if (totalSize > 0) {
                            val pct = (downloaded * 100 / totalSize).toInt()
                            listener?.onModelLoading("Downloading: $pct% (${downloaded / 1024 / 1024}MB)")
                        }
                    }
                }
            }

            // Atomic rename — prevents corrupt partial files
            tmpFile.renameTo(modelFile)
        } catch (e: Exception) {
            tmpFile.delete()
            throw e
        }
    }

    /**
     * Process raw 16kHz mono PCM audio and detect speaker.
     */
    suspend fun processAudio(audio: FloatArray) = withContext(Dispatchers.Default) {
        val sess = session
        if (sess == null) {
            listener?.onNoSpeech("MODEL NOT LOADED")
            return@withContext
        }

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
        if (rms < 0.001f) { // Very low threshold — catches distant speech
            listener?.onNoSpeech("dB:${(20 * kotlin.math.log10(rms.coerceAtLeast(1e-10f))).toInt()} (silence)")
            return@withContext
        }

        try {
            // Run ONNX model — input shape: [1, numFrames, 80]
            val inputData = Array(1) { Array(numFrames) { f ->
                FloatArray(80) { b -> features[f * 80 + b] }
            }}
            val inputTensor = OnnxTensor.createTensor(env, inputData)

            val results = sess.run(mapOf(inputName to inputTensor))
            val outputTensor = results[0] as OnnxTensor

            // Extract 256-dim embedding
            val embedding = extractEmbedding(outputTensor)

            inputTensor.close()
            results.close()

            if (embedding == null || embedding.size != EMBEDDING_DIM) {
                listener?.onNoSpeech("Bad embedding: size=${embedding?.size}")
                return@withContext
            }

            // L2 normalize the embedding
            val norm = sqrt(embedding.sumOf { (it * it).toDouble() }.toFloat())
            if (norm > 0) for (i in embedding.indices) embedding[i] /= norm

            // Compare against profiles
            matchSpeaker(embedding, rms)

        } catch (e: Exception) {
            listener?.onNoSpeech("ONNX error: ${e.message?.take(60)}")
        }
    }

    private fun extractEmbedding(tensor: OnnxTensor): FloatArray? {
        return try {
            val rawOutput = tensor.value
            when (rawOutput) {
                is Array<*> -> {
                    val first = rawOutput[0]
                    when (first) {
                        is FloatArray -> first
                        is Array<*> -> (first as? Array<FloatArray>)?.get(0)
                        else -> null
                    }
                }
                is FloatArray -> rawOutput
                else -> null
            }
        } catch (e: Exception) {
            null
        }
    }

    private fun matchSpeaker(embedding: FloatArray, rms: Float) {
        val db = (20 * kotlin.math.log10(rms.coerceAtLeast(1e-10f))).toInt()
        val dbStr = "${db}dB"

        if (profiles.isEmpty()) {
            // First speaker
            profiles.add(SpeakerProfile(embedding.clone()).also {
                it.embeddings.add(embedding.clone())
            })
            currentSpeaker = 0
            listener?.onSpeakerDetected(0, true, 1f, "$dbStr | Speaker A registered | profiles:1")
            return
        }

        // Find best matching profile
        var bestIdx = -1
        var bestSim = -1f
        val simTexts = mutableListOf<String>()

        for (i in profiles.indices) {
            val sim = cosineSimilarity(embedding, profiles[i].embedding)
            simTexts.add("${('A' + i)}=${"%.2f".format(sim)}")
            if (sim > bestSim) {
                bestSim = sim
                bestIdx = i
            }
        }

        val simsStr = simTexts.joinToString(" ")
        val thrStr = "thr=${"%.2f".format(SIMILARITY_THRESHOLD)}"

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

            val changeStr = if (isChange) ">>> CHANGED" else "same"
            listener?.onSpeakerDetected(bestIdx, isChange, bestSim,
                "$dbStr | $simsStr | $thrStr | $changeStr")
        } else {
            // New speaker
            val newIdx = profiles.size
            profiles.add(SpeakerProfile(embedding.clone()).also {
                it.embeddings.add(embedding.clone())
            })
            currentSpeaker = newIdx
            listener?.onSpeakerDetected(newIdx, true, bestSim,
                "$dbStr | $simsStr | $thrStr | NEW Speaker ${('A' + newIdx)}")
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
        modelLoaded = false
    }

    val totalSpeakers: Int get() = profiles.size
}
