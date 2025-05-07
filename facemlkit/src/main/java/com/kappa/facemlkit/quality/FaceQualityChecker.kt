package com.kappa.facemlkit.quality

import android.graphics.Bitmap
import android.util.Log
import com.google.mlkit.vision.face.Face
import com.kappa.facemlkit.models.FaceQualityResult
import com.kappa.facemlkit.models.QualityIssue
import com.kappa.facemlkit.utils.ImageUtils
import kotlin.math.abs

/**
 * Utility class for checking face quality using both Standard Laplacian and Modified Laplacian
 * Internal class not exposed directly to SDK users
 */
internal class FaceQualityChecker {

    private val TAG = "FaceQualityChecker"
    private val LAPLACIAN1_THRESHOLD = 130.0 // Standard Laplacian threshold
    private val LAPLACIAN2_THRESHOLD = 7.0   // Modified Laplacian threshold (experimental)

    fun checkFaceQuality(face: Face, bitmap: Bitmap): FaceQualityResult {
        val issues = mutableListOf<QualityIssue>()
        var failureReason: String? = null
        var qualityScore = 1.0f

        val box = face.boundingBox
        val faceBitmap = ImageUtils.cropFaceTightly(bitmap, box)
        if (faceBitmap != null) {
            // Calculate both sharpness metrics
            val laplacian1 = computeLaplacian1Sharpness(faceBitmap)
            val laplacian2 = computeLaplacian2Sharpness(faceBitmap)

            // Log both values for comparison
            Log.d(TAG, "Face sharpness values - Laplacian1: $laplacian1 (threshold: $LAPLACIAN1_THRESHOLD), " +
                    "Laplacian2: $laplacian2 (threshold: $LAPLACIAN2_THRESHOLD)")

            // Only use Laplacian1 for actual quality check
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
                        val pixel = gray[y + i][x + j]
                        val k = kernel[i + 1][j + 1]
                        sum += k * pixel
                    }
                }
                laplacian[y][x] = sum
            }
        }

        // Flatten and calculate variance directly
        val flatValues = laplacian.flatMap { it.asList() }
        val variance = flatValues.map { it * it }.average()

        return variance
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
                        val pixel = gray[y + i][x + j]
                        sumX += kernelX[i + 1][j + 1] * pixel
                        sumY += kernelY[i + 1][j + 1] * pixel
                    }
                }
                lapX[y][x] = sumX
                lapY[y][x] = sumY
            }
        }

        var mlapSum = 0.0
        for (y in 0 until height) {
            for (x in 0 until width) {
                mlapSum += abs(lapX[y][x]) + abs(lapY[y][x])
            }
        }

        return mlapSum / (width * height)
    }
}