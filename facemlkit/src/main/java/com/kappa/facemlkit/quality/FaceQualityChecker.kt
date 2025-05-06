package com.kappa.facemlkit.quality

import android.graphics.Bitmap
import android.util.Log
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceLandmark
import com.kappa.facemlkit.models.FaceQualityResult
import com.kappa.facemlkit.models.QualityIssue
import com.kappa.facemlkit.utils.ImageUtils
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Utility class for checking face quality with stricter requirements
 * Internal class not exposed directly to SDK users
 */
internal class FaceQualityChecker {

    private val TAG = "FaceQualityChecker"

    // Stricter thresholds
    private val MIN_FACE_SIZE_RATIO = 0.15f
    private val MAX_HEAD_ANGLE = 10.0f
    private val MIN_EYE_OPEN_PROB = 0.7f
    private val MIN_SHARPNESS_THRESHOLD = 100.0    // Adjusted for Laplacian

    fun checkFaceQuality(face: Face, bitmap: Bitmap): FaceQualityResult {
        val issues = mutableListOf<QualityIssue>()
        var failureReason: String? = null
        var qualityScore = 1.0f

        // Face size ratio
        val box = face.boundingBox
        val faceArea = box.width() * box.height()
        val imageArea = bitmap.width * bitmap.height
        val faceSizeRatio = faceArea.toFloat() / imageArea

        if (faceSizeRatio < MIN_FACE_SIZE_RATIO) {
            Log.d(TAG, "Quality check failed: Face size ratio ($faceSizeRatio) is too small")
            issues.add(QualityIssue.TOO_SMALL)
            failureReason = "Face is too small"
            qualityScore -= 0.3f
        }

        // Face orientation
        val angleY = face.headEulerAngleY
        val angleZ = face.headEulerAngleZ
        if (abs(angleY) > MAX_HEAD_ANGLE || abs(angleZ) > MAX_HEAD_ANGLE) {
            Log.d(TAG, "Quality check failed: Head orientation (Y=$angleY, Z=$angleZ)")
            issues.add(QualityIssue.FACE_NOT_FRONT_FACING)
            failureReason = "Face is not properly front-facing"
            qualityScore -= 0.25f
        }

        // Eyes open
        val leftEyeOpenProb = face.leftEyeOpenProbability
        val rightEyeOpenProb = face.rightEyeOpenProbability
        if (leftEyeOpenProb == null || rightEyeOpenProb == null ||
            leftEyeOpenProb < MIN_EYE_OPEN_PROB || rightEyeOpenProb < MIN_EYE_OPEN_PROB) {
            Log.d(TAG, "Quality check failed: Eyes not fully open (Left=$leftEyeOpenProb, Right=$rightEyeOpenProb)")
            issues.add(QualityIssue.EYES_CLOSED)
            failureReason = "Eyes are not fully open"
            qualityScore -= 0.2f
        }

        // Facial landmarks
        val leftEye = face.getLandmark(FaceLandmark.LEFT_EYE)
        val rightEye = face.getLandmark(FaceLandmark.RIGHT_EYE)
        val noseBase = face.getLandmark(FaceLandmark.NOSE_BASE)
        val mouthBottom = face.getLandmark(FaceLandmark.MOUTH_BOTTOM)
        val mouthLeft = face.getLandmark(FaceLandmark.MOUTH_LEFT)
        val mouthRight = face.getLandmark(FaceLandmark.MOUTH_RIGHT)

        if (leftEye == null || rightEye == null || noseBase == null ||
            mouthBottom == null || mouthLeft == null || mouthRight == null) {
            Log.d(TAG, "Quality check failed: Incomplete facial landmarks")
            issues.add(QualityIssue.KEY_LANDMARKS_MISSING)
            failureReason = "Complete facial features not detected"
            qualityScore -= 0.25f
        }

        // Face centered
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
            Log.d(TAG, "Quality check failed: Face not centered")
            issues.add(QualityIssue.FACE_NOT_CENTERED)
            failureReason = "Face is not centered"
            qualityScore -= 0.2f
        }

        // Blur detection with Laplacian
        val faceBitmap = ImageUtils.cropFaceTightly(bitmap, box)
        if (faceBitmap != null) {
            val sharpness = computeSobelSharpness(faceBitmap)
            if (sharpness < MIN_SHARPNESS_THRESHOLD) {
                Log.d(TAG, "Quality check failed: Image is blurry (sharpness=$sharpness)")
                issues.add(QualityIssue.BLURRY_FACE)
                failureReason = "Image is too blurry"
                qualityScore -= 0.25f
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
     * Computes blur using Laplacian variance (name kept as computeSobelSharpness)
     */
    private fun computeSobelSharpness(bitmap: Bitmap): Double {
        val width = bitmap.width
        val height = bitmap.height

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

        val kernel = arrayOf(
            intArrayOf(0, 1, 0),
            intArrayOf(1, -4, 1),
            intArrayOf(0, 1, 0)
        )

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

        val flatValues = laplacian.flatMap { it.asList() }
        val mean = flatValues.average()
        return flatValues.map { (it - mean) * (it - mean) }.average()

    }
}
