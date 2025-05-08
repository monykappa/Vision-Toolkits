package com.kappa.facemlkit.quality

import android.graphics.Bitmap
import android.util.Log
import com.google.mlkit.vision.face.Face
import com.kappa.facemlkit.models.FaceQualityResult
import com.kappa.facemlkit.models.QualityIssue
import com.kappa.facemlkit.utils.ImageUtils
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

/**
 * Utility class for checking face quality using both Standard Laplacian and Modified Laplacian
 * Internal class not exposed directly to SDK users
 */
internal class FaceQualityChecker {

    private val TAG = "FaceQualityChecker"
    private val LAPLACIAN1_THRESHOLD = 150.0 // Standard Laplacian threshold
    private val LAPLACIAN2_THRESHOLD = 7.0   // Modified Laplacian threshold (experimental)

    /**
     * Performs Adaptive Gamma Correction with Weighting Distribution (AGCWD) on a grayscale image.
     * @param gray Array of grayscale pixel values
     * @param width Image width
     * @param height Image height
     * @param a Parameter controlling the enhancement (default 0.25)
     * @param truncatedCdf Whether to truncate CDF (default false)
     * @return Processed grayscale array
     */
    private fun imageAgcwd(gray: Array<DoubleArray>, width: Int, height: Int, a: Double = 0.25, truncatedCdf: Boolean = false): Array<DoubleArray> {
        // Compute histogram
        val hist = IntArray(256)
        for (y in 0 until height) {
            for (x in 0 until width) {
                hist[gray[y][x].toInt()]++
            }
        }

        // Compute CDF
        val cdf = hist.scan(0, Int::plus)
        val cdfMax = cdf.maxOrNull() ?: 1
        val cdfNormalized = cdf.map { it.toDouble() / cdfMax }.toDoubleArray()

        // Compute normalized probabilities
        val probNormalized = hist.map { it.toDouble() / (width * height) }.toDoubleArray()
        val probMin = probNormalized.minOrNull() ?: 0.0
        val probMax = probNormalized.maxOrNull() ?: 1.0

        // Apply weighting distribution
        val pnTemp = probNormalized.map { (it - probMin) / (probMax - probMin) }.toDoubleArray()
        for (i in pnTemp.indices) {
            pnTemp[i] = when {
                pnTemp[i] > 0 -> probMax * pnTemp[i].pow(a)
                pnTemp[i] < 0 -> probMax * (-((-pnTemp[i]).pow(a)))
                else -> 0.0
            }
        }
        val pnSum = pnTemp.sum()
        val probNormalizedWd = pnTemp.map { it / pnSum }.toDoubleArray()

        // Compute weighted CDF
        val cdfProbNormalizedWd = probNormalizedWd.scan(0.0, Double::plus)
        val inverseCdf = if (truncatedCdf) {
            cdfProbNormalizedWd.map { max(0.5, 1.0 - it) }.toDoubleArray()
        } else {
            cdfProbNormalizedWd.map { 1.0 - it }.toDoubleArray()
        }

        // Apply transformation
        val result = Array(height) { DoubleArray(width) }
        for (y in 0 until height) {
            for (x in 0 until width) {
                val intensity = gray[y][x].toInt()
                result[y][x] = (255 * (intensity / 255.0).pow(inverseCdf[intensity])).toInt().toDouble()
            }
        }
        return result
    }

    /**
     * Processes bright images by inverting, applying AGCWD, and inverting back.
     */
    private fun processBright(gray: Array<DoubleArray>, width: Int, height: Int): Array<DoubleArray> {
        val negative = Array(height) { y -> DoubleArray(width) { x -> 255.0 - gray[y][x] } }
        val agcwd = imageAgcwd(negative, width, height, a = 0.25, truncatedCdf = false)
        return Array(height) { y -> DoubleArray(width) { x -> 255.0 - agcwd[y][x] } }
    }

    /**
     * Processes dimmed images with AGCWD.
     */
    private fun processDimmed(gray: Array<DoubleArray>, width: Int, height: Int): Array<DoubleArray> {
        return imageAgcwd(gray, width, height, a = 0.55, truncatedCdf = true)
    }

    /**
     * Adjusts brightness adaptively based on image mean intensity.
     * @param src Input bitmap
     * @param brightnessOffset Ignored in this adaptive implementation
     * @return Processed bitmap
     */
    private fun adjustBrightness(src: Bitmap, brightnessOffset: Int): Bitmap {
        val width = src.width
        val height = src.height
        val bmp = src.copy(src.config, true)

        // Convert to grayscale array
        val gray = Array(height) { DoubleArray(width) }
        var meanIntensity = 0.0
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = src.getPixel(x, y)
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                gray[y][x] = 0.299 * r + 0.587 * g + 0.114 * b
                meanIntensity += gray[y][x]
            }
        }
        meanIntensity /= (width * height)

        // Determine processing based on mean intensity
        val threshold = 0.2
        val expIn = 112.0
        val t = (meanIntensity - expIn) / expIn
        val processedGray = when {
            t < -threshold -> {
                Log.d(TAG, "Applying dimmed image processing")
                processDimmed(gray, width, height)
            }
            t > threshold -> {
                Log.d(TAG, "Applying bright image processing")
                processBright(gray, width, height)
            }
            else -> {
                Log.d(TAG, "No brightness adjustment needed")
                gray
            }
        }

        // Convert back to bitmap
        for (y in 0 until height) {
            for (x in 0 until width) {
                val value = processedGray[y][x].toInt().coerceIn(0, 255)
                val a = (src.getPixel(x, y) ushr 24) and 0xFF
                bmp.setPixel(x, y, (a shl 24) or (value shl 16) or (value shl 8) or value)
            }
        }
        return bmp
    }

    /**
     * Main entry point: crops, optionally brightens, and checks sharpness using Laplacian.
     *
     * @param brightnessOffset adjustment to apply before sharpness checks (default 0)
     */
    fun checkFaceQuality(
        face: Face,
        bitmap: Bitmap,
        brightnessOffset: Int = 0
    ): FaceQualityResult {
        val issues = mutableListOf<QualityIssue>()
        var failureReason: String? = null
        var qualityScore = 1.0f

        // Crop the face region
        val box = face.boundingBox
        val faceBitmap = ImageUtils.cropFaceTightly(bitmap, box)

        if (faceBitmap != null) {
            // 1) adjust brightness if requested
            val processedBitmap = if (brightnessOffset != 0) {
                adjustBrightness(faceBitmap, brightnessOffset)
            } else {
                faceBitmap
            }

            // 2) compute both sharpness metrics
            val laplacian1 = computeLaplacian1Sharpness(processedBitmap)
            val laplacian2 = computeLaplacian2Sharpness(processedBitmap)

            Log.d(TAG, "After brightness($brightnessOffset): Lap1=$laplacian1 (thr=$LAPLACIAN1_THRESHOLD), " +
                    "Lap2=$laplacian2 (thr=$LAPLACIAN2_THRESHOLD)")

            // Use standard Laplacian1 for pass/fail
            if (laplacian1 < LAPLACIAN1_THRESHOLD) {
                Log.d(TAG, "Quality check failed: Image is blurry (Laplacian1=$laplacian1)")
                issues.add(QualityIssue.BLURRY_FACE)
                failureReason = "Image is too blurry"
                qualityScore -= 0.25f
            } else {
                Log.d(TAG, "Quality check passed: Image is sharp (Laplacian1=$laplacian1)")
            }
        } else {
            Log.d(TAG, "Unable to crop face for blur detection")
            issues.add(QualityIssue.BLURRY_FACE)
            failureReason = "Unable to process face for quality check"
            qualityScore -= 0.25f
        }

        // Ensure score within [0,1]
        qualityScore = qualityScore.coerceIn(0.0f, 1.0f)

        return FaceQualityResult(
            isGoodQuality = issues.isEmpty(),
            qualityScore = qualityScore,
            issues = issues,
            failureReason = failureReason
        )
    }

    /**
     * Standard Laplacian-based sharpness calculation (primary method)
     */
    private fun computeLaplacian1Sharpness(bitmap: Bitmap): Double {
        val width = bitmap.width
        val height = bitmap.height

        // Convert to grayscale
        val gray = Array(height) { DoubleArray(width) }
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = bitmap.getPixel(x, y)
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                gray[y][x] = 0.299 * r + 0.587 * g + 0.114 * b
            }
        }

        // Laplacian kernel
        val kernel = arrayOf(
            intArrayOf(0, 1, 0),
            intArrayOf(1, -4, 1),
            intArrayOf(0, 1, 0)
        )

        // Apply kernel
        val laplacian = Array(height) { DoubleArray(width) }
        for (y in 1 until height - 1) {
            for (x in 1 until width - 1) {
                var sum = 0.0
                for (i in -1..1) {
                    for (j in -1..1) {
                        sum += kernel[i + 1][j + 1] * gray[y + i][x + j]
                    }
                }
                laplacian[y][x] = sum
            }
        }

        // Calculate variance (mean squared value)
        val flatValues = laplacian.flatMap { it.asList() }
        return flatValues.map { it * it }.average()
    }

    /**
     * Modified Laplacian-based sharpness calculation (experimental method)
     */
    private fun computeLaplacian2Sharpness(bitmap: Bitmap): Double {
        val width = bitmap.width
        val height = bitmap.height

        // Convert to grayscale
        val gray = Array(height) { DoubleArray(width) }
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = bitmap.getPixel(x, y)
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                gray[y][x] = 0.299 * r + 0.587 * g + 0.114 * b
            }
        }

        // Kernels for X and Y directions
        val kernelX = arrayOf(
            doubleArrayOf(0.0, 0.0, 0.0),
            doubleArrayOf(-1.0, 2.0, -1.0),
            doubleArrayOf(0.0, 0.0, 0.0)
        )
        val kernelY = arrayOf(
            doubleArrayOf(0.0, -1.0, 0.0),
            doubleArrayOf(0.0, 2.0, 0.0),
            doubleArrayOf(0.0, -1.0, 0.0)
        )

        val lapX = Array(height) { DoubleArray(width) }
        val lapY = Array(height) { DoubleArray(width) }

        for (y in 1 until height - 1) {
            for (x in 1 until width - 1) {
                var sumX = 0.0
                var sumY = 0.0
                for (i in -1..1) {
                    for (j in -1..1) {
                        sumX += kernelX[i + 1][j + 1] * gray[y + i][x + j]
                        sumY += kernelY[i + 1][j + 1] * gray[y + i][x + j]
                    }
                }
                lapX[y][x] = sumX
                lapY[y][x] = sumY
            }
        }

        // Compute mean absolute sum of both directions
        var mlapSum = 0.0
        for (y in 0 until height) {
            for (x in 0 until width) {
                mlapSum += abs(lapX[y][x]) + abs(lapY[y][x])
            }
        }
        return mlapSum / (width * height)
    }
}