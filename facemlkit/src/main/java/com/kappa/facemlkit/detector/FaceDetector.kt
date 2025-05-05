package com.kappa.facemlkit.detector

import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.kappa.facemlkit.models.FaceQualityResult
import com.kappa.facemlkit.quality.FaceQualityChecker
import com.kappa.facemlkit.utils.ImageUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.tasks.await
import kotlinx.coroutines.withContext

/**
 * Core face detection implementation using ML Kit
 * Internal class not exposed directly to SDK users
 */
internal class FaceDetector {

    private val TAG = "FaceDetector"
    private val faceQualityChecker = FaceQualityChecker()

    private val faceDetectorOptions = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
        .setMinFaceSize(0.15f)
        .enableTracking()
        .build()

    private val detector = FaceDetection.getClient(faceDetectorOptions)

    /**
     * Result object for face extraction operations
     */
    data class FaceExtractionResult(
        val faceBitmap: Bitmap?,
        val qualityResult: FaceQualityResult
    )

    /**
     * Detect faces in the input image
     *
     * @param bitmap Input image
     * @return List of detected faces
     */
    suspend fun detectFaces(bitmap: Bitmap): List<Face> = withContext(Dispatchers.IO) {
        try {
            val image = InputImage.fromBitmap(bitmap, 0)
            Log.d(TAG, "Detecting faces in image")
            val faces = detector.process(image).await()
            Log.d(TAG, "Detected ${faces.size} faces in the image")
            faces
        } catch (e: Exception) {
            Log.e(TAG, "Error detecting faces: ${e.message}", e)
            emptyList()
        }
    }

    /**
     * Extract the largest face from the input image
     *
     * @param bitmap Input image
     * @return Processed face bitmap and quality assessment
     */
    suspend fun extractLargestFace(bitmap: Bitmap): FaceExtractionResult = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Loaded image size: ${bitmap.width}x${bitmap.height}")
            var faces = detectFaces(bitmap)

            // If no faces found, try compressed version
            if (faces.isEmpty()) {
                Log.d(TAG, "No face found at original quality, trying with compressed image")
                val compressedBitmap = ImageUtils.compressBitmap(bitmap, 80)
                faces = detectFaces(compressedBitmap)
            }

            if (faces.isEmpty()) {
                Log.d(TAG, "No faces detected, using fallback resize and crop")
                // Create a quality result indicating no face detected
                val noFaceQualityResult = FaceQualityResult(
                    isGoodQuality = false,
                    qualityScore = 0.0f,
                    issues = listOf(com.kappa.facemlkit.models.QualityIssue.NO_FACE_DETECTED),
                    failureReason = "No face detected in the image"
                )

                // Apply fallback processing to the input image
                val fallbackBitmap = ImageUtils.applyResizeAndCenterCrop(bitmap)
                return@withContext FaceExtractionResult(
                    faceBitmap = fallbackBitmap,
                    qualityResult = noFaceQualityResult
                )
            }

            val largestFace = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }
                ?: return@withContext FaceExtractionResult(
                    faceBitmap = ImageUtils.applyResizeAndCenterCrop(bitmap),
                    qualityResult = FaceQualityResult(
                        isGoodQuality = false,
                        qualityScore = 0.0f,
                        issues = listOf(com.kappa.facemlkit.models.QualityIssue.NO_FACE_DETECTED),
                        failureReason = "No face detected in the image"
                    )
                )

            // Perform comprehensive quality check
            val qualityResult = faceQualityChecker.checkFaceQuality(largestFace, bitmap)
            Log.d(TAG, "Face quality check result: isGoodQuality=${qualityResult.isGoodQuality}, score=${qualityResult.qualityScore}")

            // Process the face bitmap regardless of quality
            val box = largestFace.boundingBox
            val faceBitmap = ImageUtils.cropFaceTightly(bitmap, box)
            val processedBitmap = faceBitmap?.let { ImageUtils.applyResizeAndCenterCrop(it) }
                ?: ImageUtils.applyResizeAndCenterCrop(bitmap)

            return@withContext FaceExtractionResult(
                faceBitmap = processedBitmap,
                qualityResult = qualityResult
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error extracting largest face: ${e.message}", e)

            // On error, apply the fallback approach
            try {
                val fallbackBitmap = ImageUtils.applyResizeAndCenterCrop(bitmap)
                return@withContext FaceExtractionResult(
                    faceBitmap = fallbackBitmap,
                    qualityResult = FaceQualityResult(
                        isGoodQuality = false,
                        qualityScore = 0.0f,
                        issues = listOf(com.kappa.facemlkit.models.QualityIssue.BLURRY_FACE),
                        failureReason = "Error processing image: ${e.message}"
                    )
                )
            } catch (e2: Exception) {
                Log.e(TAG, "Fallback also failed: ${e2.message}", e2)
                return@withContext FaceExtractionResult(
                    faceBitmap = null,
                    qualityResult = FaceQualityResult(
                        isGoodQuality = false,
                        qualityScore = 0.0f,
                        issues = listOf(com.kappa.facemlkit.models.QualityIssue.BLURRY_FACE),
                        failureReason = "Failed to process image"
                    )
                )
            }
        }
    }

    /**
     * Check the quality of the largest face in an image
     *
     * @param bitmap Input image
     * @return Face quality assessment result
     */
    suspend fun assessFaceQuality(bitmap: Bitmap): FaceQualityResult = withContext(Dispatchers.IO) {
        try {
            val faces = detectFaces(bitmap)

            if (faces.isEmpty()) {
                return@withContext FaceQualityResult(
                    isGoodQuality = false,
                    qualityScore = 0.0f,
                    issues = listOf(com.kappa.facemlkit.models.QualityIssue.NO_FACE_DETECTED),
                    failureReason = "No face detected in the image"
                )
            }

            val largestFace = faces.maxByOrNull {
                it.boundingBox.width() * it.boundingBox.height()
            } ?: return@withContext FaceQualityResult(
                isGoodQuality = false,
                qualityScore = 0.0f,
                issues = listOf(com.kappa.facemlkit.models.QualityIssue.NO_FACE_DETECTED),
                failureReason = "No face detected in the image"
            )

            faceQualityChecker.checkFaceQuality(largestFace, bitmap)
        } catch (e: Exception) {
            Log.e(TAG, "Error assessing face quality: ${e.message}", e)
            FaceQualityResult(
                isGoodQuality = false,
                qualityScore = 0.0f,
                issues = listOf(com.kappa.facemlkit.models.QualityIssue.BLURRY_FACE),
                failureReason = "Error during quality assessment: ${e.message}"
            )
        }
    }

    /**
     * Release ML Kit detector
     */
    fun close() {
        detector.close()
    }
}