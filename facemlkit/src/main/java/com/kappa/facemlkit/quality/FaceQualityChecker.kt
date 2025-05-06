package com.kappa.facemlkit.quality

import android.graphics.Bitmap
import android.util.Log
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceLandmark
import com.kappa.facemlkit.models.FaceQualityResult
import com.kappa.facemlkit.models.QualityIssue
import com.kappa.facemlkit.utils.ImageUtils
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Utility class for checking face quality with stricter requirements
 * Internal class not exposed directly to SDK users
 */
internal class FaceQualityChecker {

    private val TAG = "FaceQualityChecker"

    // Configurable thresholds for quality checks - increased strictness
    private val MIN_FACE_SIZE_RATIO = 0.15f       // Increased from 0.10f
    private val MAX_HEAD_ANGLE = 10.0f            // Decreased from 15.0f
    private val MIN_EYE_OPEN_PROB = 0.7f          // Increased from 0.l
    private val MIN_SHARPNESS_THRESHOLD = 24.0    // Increased from 20.0

    /**ll
     * Checks comprehensive quality metrics for the face with stricter requirements
     *
     * @param face ML Kit face object
     * @param bitmap Original image containing the face
     * @return Face quality result with details
     */
    fun checkFaceQuality(face: Face, bitmap: Bitmap): FaceQualityResult {
        val issues = mutableListOf<QualityIssue>()
        var failureReason: String? = null
        var qualityScore = 1.0f

        // Check face size ratio (STRICTER)
        val box = face.boundingBox
        val faceArea = box.width() * box.height()
        val imageArea = bitmap.width * bitmap.height
        val faceSizeRatio = faceArea.toFloat() / imageArea

        if (faceSizeRatio < MIN_FACE_SIZE_RATIO) {
            Log.d(TAG, "Quality check failed: Face size ratio ($faceSizeRatio) is too small (< $MIN_FACE_SIZE_RATIO)")
            issues.add(QualityIssue.TOO_SMALL)
            failureReason = "Face is too small"
            qualityScore -= 0.3f
        }
        Log.d(TAG, "Face size ratio: $faceSizeRatio")

        // Check face orientation (STRICTER)
        val angleY = face.headEulerAngleY
        val angleZ = face.headEulerAngleZ
        if (abs(angleY) > MAX_HEAD_ANGLE || abs(angleZ) > MAX_HEAD_ANGLE) {
            Log.d(TAG, "Quality check failed: Head orientation (Y=$angleY, Z=$angleZ) is not properly front-facing")
            issues.add(QualityIssue.FACE_NOT_FRONT_FACING)
            failureReason = "Face is not properly front-facing"
            qualityScore -= 0.25f
        }
        Log.d(TAG, "Head orientation: Y=$angleY, Z=$angleZ")

        // Check if eyes are open (STRICTER)
        val leftEyeOpenProb = face.leftEyeOpenProbability
        val rightEyeOpenProb = face.rightEyeOpenProbability
        if (leftEyeOpenProb == null || rightEyeOpenProb == null ||
            leftEyeOpenProb < MIN_EYE_OPEN_PROB || rightEyeOpenProb < MIN_EYE_OPEN_PROB) {
            Log.d(TAG, "Quality check failed: Eyes are not fully open (Left=$leftEyeOpenProb, Right=$rightEyeOpenProb)")
            issues.add(QualityIssue.EYES_CLOSED)
            failureReason = "Eyes are not fully open"
            qualityScore -= 0.2f
        }
        Log.d(TAG, "Eye openness: Left=$leftEyeOpenProb, Right=$rightEyeOpenProb")

        // Check for key landmarks (MORE COMPREHENSIVE)
        val leftEye = face.getLandmark(FaceLandmark.LEFT_EYE)
        val rightEye = face.getLandmark(FaceLandmark.RIGHT_EYE)
        val noseBase = face.getLandmark(FaceLandmark.NOSE_BASE)
        val mouthBottom = face.getLandmark(FaceLandmark.MOUTH_BOTTOM)
        val mouthLeft = face.getLandmark(FaceLandmark.MOUTH_LEFT)
        val mouthRight = face.getLandmark(FaceLandmark.MOUTH_RIGHT)

        if (leftEye == null || rightEye == null || noseBase == null ||
            mouthBottom == null || mouthLeft == null || mouthRight == null) {
            Log.d(TAG, "Quality check failed: Complete facial landmarks not detected")
            issues.add(QualityIssue.KEY_LANDMARKS_MISSING)
            failureReason = "Complete facial features not detected"
            qualityScore -= 0.25f
        }
        Log.d(TAG, "Complete landmarks detected: ${leftEye != null && rightEye != null && noseBase != null && mouthBottom != null}")

        // Check face is centered in the image
        val centerX = bitmap.width / 2
        val centerY = bitmap.height / 2
        val faceCenter = face.boundingBox.let {
            Pair(it.exactCenterX().toInt(), it.exactCenterY().toInt())
        }
        val distanceFromCenter = sqrt(
            (centerX - faceCenter.first).toDouble().pow(2) +
                    (centerY - faceCenter.second).toDouble().pow(2)
        )
        val maxAllowedDistance = minOf(bitmap.width, bitmap.height) * 0.15

        if (distanceFromCenter > maxAllowedDistance) {
            Log.d(TAG, "Quality check failed: Face is not centered in the image")
            issues.add(QualityIssue.FACE_NOT_CENTERED)
            failureReason = "Face is not centered in the image"
            qualityScore -= 0.2f
        }

        // Check image sharpness (STRICTER)
        val faceBitmap = ImageUtils.cropFaceTightly(bitmap, box)
        if (faceBitmap != null) {
            val sharpness = computeSobelSharpness(faceBitmap)
            val sharpnessThreshold = MIN_SHARPNESS_THRESHOLD
            if (sharpness < sharpnessThreshold) {
                Log.d(TAG, "Quality check failed: Image is blurry (Sobel sharpness=$sharpness, threshold=$sharpnessThreshold)")
                issues.add(QualityIssue.BLURRY_FACE)
                failureReason = "Image is too blurry"
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

    /**
     * Extension function for Double.pow
     */
    private fun Double.pow(exponent: Int): Double = Math.pow(this, exponent.toDouble())
}
