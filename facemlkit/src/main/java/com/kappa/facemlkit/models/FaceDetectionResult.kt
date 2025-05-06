package com.kappa.facemlkit.models

import com.google.mlkit.vision.face.Face

/**
 * Result object for face detection operations
 *
 * @property success Whether the operation was successful
 * @property faces List of detected faces
 * @property message Human-readable result message
 */
data class FaceDetectionResult(
    val success: Boolean,
    val faces: List<Face>,
    val message: String
)

/**
 * Result object for face quality assessment
 *
 * @property isGoodQuality Whether the face meets quality criteria
 * @property qualityScore Quality score between 0.0 and 1.0
 * @property issues List of detected quality issues
 * @property failureReason Human-readable failure reason
 */
data class FaceQualityResult(
    val isGoodQuality: Boolean,
    val qualityScore: Float,
    val issues: List<QualityIssue> = emptyList(),
    val failureReason: String? = null
)

/**
 * Enum representing possible face quality issues
 */
enum class QualityIssue {
    TOO_SMALL,
    EYES_CLOSED,
    FACE_TOO_DARK,
//    FACE_NOT_CENTERED,
    MULTIPLE_FACES,
    NO_FACE_DETECTED,
    BLURRY_FACE,
    FACE_NOT_FRONT_FACING,
    KEY_LANDMARKS_MISSING
}