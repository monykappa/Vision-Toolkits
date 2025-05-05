package com.kappa.facemlkit.quality

import android.graphics.Bitmap
import android.util.Log
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceLandmark
import com.kappa.facemlkit.models.FaceQualityResult
import com.kappa.facemlkit.models.QualityIssue
import com.kappa.facemlkit.utils.ImageUtils
import kotlin.math.sqrt

/**
 * Utility class for checking face quality
 * Internal class not exposed directly to SDK users
 */
internal class FaceQualityChecker {

    private val TAG = "FaceQualityChecker"

    /**
     * Checks comprehensive quality metrics for the face
     *
     * @param face ML Kit face object
     * @param bitmap Original image containing the face
     * @return Face quality result with details
     */
    fun checkFaceQuality(face: Face, bitmap: Bitmap): FaceQualityResult {
        val issues = mutableListOf<QualityIssue>()
        var failureReason: String? = null
        var qualityScore = 1.0f

        // Check face size ratio
        val box = face.boundingBox
        val faceArea = box.width() * box.height()
        val imageArea = bitmap.width * bitmap.height
        val faceSizeRatio = faceArea.toFloat() / imageArea

        if (faceSizeRatio < 0.10f) {
            Log.d(TAG, "Quality check failed: Face size ratio ($faceSizeRatio) is too small (< 0.10)")
            issues.add(QualityIssue.TOO_SMALL)
            failureReason = "Face is too small"
            qualityScore -= 0.3f
        }
        Log.d(TAG, "Face size ratio: $faceSizeRatio")

        // Check face orientation
        val angleY = face.headEulerAngleY
        val angleZ = face.headEulerAngleZ
        if (kotlin.math.abs(angleY) > 15 || kotlin.math.abs(angleZ) > 15) {
            Log.d(TAG, "Quality check failed: Head orientation (Y=$angleY, Z=$angleZ) is not front-facing")
            issues.add(QualityIssue.FACE_NOT_FRONT_FACING)
            failureReason = "Face is not front-facing"
            qualityScore -= 0.25f
        }
        Log.d(TAG, "Head orientation: Y=$angleY, Z=$angleZ")

        // Check if eyes are open
        val leftEyeOpenProb = face.leftEyeOpenProbability
        val rightEyeOpenProb = face.rightEyeOpenProbability
        if (leftEyeOpenProb == null || rightEyeOpenProb == null || leftEyeOpenProb < 0.5f || rightEyeOpenProb < 0.5f) {
            Log.d(TAG, "Quality check failed: Eyes are not open (Left=$leftEyeOpenProb, Right=$rightEyeOpenProb)")
            issues.add(QualityIssue.EYES_CLOSED)
            failureReason = "Eyes are not open"
            qualityScore -= 0.2f
        }
        Log.d(TAG, "Eye openness: Left=$leftEyeOpenProb, Right=$rightEyeOpenProb")

        // Check for key landmarks
        val leftEye = face.getLandmark(FaceLandmark.LEFT_EYE)
        val rightEye = face.getLandmark(FaceLandmark.RIGHT_EYE)
        val noseBase = face.getLandmark(FaceLandmark.NOSE_BASE)
        if (leftEye == null || rightEye == null || noseBase == null) {
            Log.d(TAG, "Quality check failed: Key facial landmarks not detected")
            issues.add(QualityIssue.KEY_LANDMARKS_MISSING)
            failureReason = "Key facial features not detected"
            qualityScore -= 0.25f
        }
        Log.d(TAG, "Key landmarks detected: ${leftEye != null && rightEye != null && noseBase != null}")

        // Check image sharpness
        val faceBitmap = ImageUtils.cropFaceTightly(bitmap, box)
        if (faceBitmap != null) {
            val sharpness = computeSobelSharpness(faceBitmap)
            val sharpnessThreshold = 20.0
            if (sharpness < sharpnessThreshold) {
                Log.d(TAG, "Quality check failed: Image is blurry (Sobel sharpness=$sharpness, threshold=$sharpnessThreshold)")
                issues.add(QualityIssue.BLURRY_FACE)
                failureReason = "Image is blurry"
                qualityScore -= 0.25f
            }
            Log.d(TAG, "Image sharpness: $sharpness (threshold=$sharpnessThreshold)")
        } else {
            Log.d(TAG, "Quality check failed: Unable to crop face for blur detection")
            issues.add(QualityIssue.BLURRY_FACE)
            failureReason = "Unable to process face for quality check"
            qualityScore -= 0.25f
        }

        // Ensure score is within valid range
        qualityScore = qualityScore.coerceIn(0.0f, 1.0f)

        val isGoodQuality = issues.isEmpty()
        Log.d(TAG, "Final quality assessment: isGoodQuality=$isGoodQuality, score=$qualityScore, issues=${issues.size}")

        return FaceQualityResult(
            isGoodQuality = isGoodQuality,
            qualityScore = qualityScore,
            issues = issues,
            failureReason = failureReason
        )
    }

    /**
     * Compute Sobel sharpness measure for blur detection
     *
     * @param bitmap Input face bitmap
     * @return Sharpness score (higher is better)
     */
    private fun computeSobelSharpness(bitmap: Bitmap): Double {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // Convert to grayscale
        val gray = Array(height) { IntArray(width) }
        for (y in 0 until height) {
            for (x in 0 until width) {
                val color = pixels[y * width + x]
                val r = (color shr 16) and 0xFF
                val g = (color shr 8) and 0xFF
                val b = color and 0xFF
                gray[y][x] = (0.299 * r + 0.587 * g + 0.114 * b).toInt()
            }
        }

        // Sobel operators
        val gx = arrayOf(
            intArrayOf(-1, 0, 1),
            intArrayOf(-2, 0, 2),
            intArrayOf(-1, 0, 1)
        )
        val gy = arrayOf(
            intArrayOf(-1, -2, -1),
            intArrayOf(0, 0, 0),
            intArrayOf(1, 2, 1)
        )

        var sum = 0.0
        var count = 0

        // Apply Sobel operator and calculate average gradient magnitude
        for (y in 1 until height - 1) {
            for (x in 1 until width - 1) {
                var sumX = 0
                var sumY = 0
                for (i in -1..1) {
                    for (j in -1..1) {
                        val pixel = gray[y + i][x + j]
                        sumX += pixel * gx[i + 1][j + 1]
                        sumY += pixel * gy[i + 1][j + 1]
                    }
                }
                val magnitude = sqrt((sumX * sumX + sumY * sumY).toDouble())
                sum += magnitude
                count++
            }
        }

        return if (count > 0) sum / count else 0.0
    }
}