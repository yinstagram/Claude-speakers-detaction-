package com.speakerdetect

import kotlin.math.*

/**
 * Computes 80-bin mel filterbank features from raw 16kHz PCM audio.
 * This matches the WeSpeaker model's expected input format:
 * - 25ms frame length (400 samples at 16kHz)
 * - 10ms frame shift (160 samples at 16kHz)
 * - 80 mel filter banks
 * - Frequency range: 20Hz to 7600Hz
 */
class FbankExtractor(
    private val sampleRate: Int = 16000,
    private val frameLength: Int = 400,   // 25ms at 16kHz
    private val frameShift: Int = 160,    // 10ms at 16kHz
    private val numMelBins: Int = 80,
    private val fMin: Float = 20f,
    private val fMax: Float = 7600f
) {
    private val fftSize = nextPowerOf2(frameLength)
    private val melFilterbank: Array<FloatArray>
    private val hammingWindow: FloatArray

    init {
        hammingWindow = FloatArray(frameLength) { i ->
            0.54f - 0.46f * cos(2.0 * PI * i / (frameLength - 1)).toFloat()
        }
        melFilterbank = buildMelFilterbank()
    }

    private fun nextPowerOf2(n: Int): Int {
        var v = n - 1
        v = v or (v shr 1)
        v = v or (v shr 2)
        v = v or (v shr 4)
        v = v or (v shr 8)
        v = v or (v shr 16)
        return v + 1
    }

    private fun hzToMel(hz: Float): Float = 2595f * log10(1f + hz / 700f)
    private fun melToHz(mel: Float): Float = 700f * (10f.pow(mel / 2595f) - 1f)

    private fun buildMelFilterbank(): Array<FloatArray> {
        val nBins = fftSize / 2 + 1
        val melMin = hzToMel(fMin)
        val melMax = hzToMel(min(fMax, sampleRate / 2f))
        val melPoints = FloatArray(numMelBins + 2) { i ->
            melToHz(melMin + (melMax - melMin) * i / (numMelBins + 1))
        }
        val binPoints = IntArray(numMelBins + 2) { i ->
            ((melPoints[i] * fftSize / sampleRate) + 0.5f).toInt()
        }

        return Array(numMelBins) { m ->
            FloatArray(nBins).also { fb ->
                for (k in binPoints[m] until min(binPoints[m + 1], nBins)) {
                    fb[k] = (k - binPoints[m]).toFloat() / max(1, binPoints[m + 1] - binPoints[m])
                }
                for (k in binPoints[m + 1] until min(binPoints[m + 2], nBins)) {
                    fb[k] = (binPoints[m + 2] - k).toFloat() / max(1, binPoints[m + 2] - binPoints[m + 1])
                }
            }
        }
    }

    /**
     * Extract fbank features from raw PCM audio (Float, -1 to 1 range).
     * Returns shape [numFrames, 80] as a flat FloatArray.
     */
    fun extract(audio: FloatArray): FloatArray {
        val numFrames = max(0, (audio.size - frameLength) / frameShift + 1)
        if (numFrames == 0) return FloatArray(0)

        val features = FloatArray(numFrames * numMelBins)
        val nBins = fftSize / 2 + 1

        for (f in 0 until numFrames) {
            val offset = f * frameShift

            // Apply Hamming window
            val frame = FloatArray(fftSize)
            for (i in 0 until frameLength) {
                frame[i] = audio[offset + i] * hammingWindow[i]
            }

            // FFT (real-valued, returns magnitude squared)
            val powerSpectrum = computePowerSpectrum(frame, nBins)

            // Apply mel filterbank
            for (m in 0 until numMelBins) {
                var energy = 0f
                for (k in 0 until nBins) {
                    energy += powerSpectrum[k] * melFilterbank[m][k]
                }
                features[f * numMelBins + m] = ln(max(energy, 1e-10f))
            }
        }

        // Apply cepstral mean normalization (CMN)
        applyCMN(features, numFrames, numMelBins)

        return features
    }

    fun getNumFrames(audioLength: Int): Int = max(0, (audioLength - frameLength) / frameShift + 1)

    private fun computePowerSpectrum(frame: FloatArray, nBins: Int): FloatArray {
        // Simple DFT for power spectrum (real input)
        val real = FloatArray(fftSize)
        val imag = FloatArray(fftSize)
        frame.copyInto(real)

        fft(real, imag)

        return FloatArray(nBins) { k ->
            real[k] * real[k] + imag[k] * imag[k]
        }
    }

    private fun fft(real: FloatArray, imag: FloatArray) {
        val n = real.size
        // Bit reversal
        var j = 0
        for (i in 0 until n - 1) {
            if (i < j) {
                var temp = real[i]; real[i] = real[j]; real[j] = temp
                temp = imag[i]; imag[i] = imag[j]; imag[j] = temp
            }
            var k = n / 2
            while (k <= j) { j -= k; k /= 2 }
            j += k
        }
        // FFT butterfly
        var len = 2
        while (len <= n) {
            val halfLen = len / 2
            val angle = -2.0 * PI / len
            for (i in 0 until n step len) {
                for (jj in 0 until halfLen) {
                    val wr = cos(angle * jj).toFloat()
                    val wi = sin(angle * jj).toFloat()
                    val tr = real[i + jj + halfLen] * wr - imag[i + jj + halfLen] * wi
                    val ti = real[i + jj + halfLen] * wi + imag[i + jj + halfLen] * wr
                    real[i + jj + halfLen] = real[i + jj] - tr
                    imag[i + jj + halfLen] = imag[i + jj] - ti
                    real[i + jj] += tr
                    imag[i + jj] += ti
                }
            }
            len *= 2
        }
    }

    private fun applyCMN(features: FloatArray, numFrames: Int, numBins: Int) {
        // Compute mean per bin across all frames
        val mean = FloatArray(numBins)
        for (f in 0 until numFrames) {
            for (b in 0 until numBins) {
                mean[b] += features[f * numBins + b]
            }
        }
        for (b in 0 until numBins) mean[b] /= numFrames

        // Subtract mean
        for (f in 0 until numFrames) {
            for (b in 0 until numBins) {
                features[f * numBins + b] -= mean[b]
            }
        }
    }
}
