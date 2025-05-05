package com.kappa.facemlkit

import android.content.Context
import android.graphics.Bitmap
import com.kappa.facemlkit.detector.FaceDetector
import com.kappa.facemlkit.models.FaceDetectionResult
import com.kappa.facemlkit.models.FaceQualityResult
import com.google.mlkit.vision.face.Face
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * FaceMLKit - Main entry point for the face detection and processing SDK
 *
 * This class provides a simplified interface for:
 * - Face detection in images
 * - Largest face extraction and preprocessing
 * - Face quality assessment
 */
class FaceMLKit private constructor(context: Context) {

    private val faceDetector = FaceDetector()
    private val coroutineScope = CoroutineScope(Dispatchers.Default)

    companion object {
        @Volatile private var instance: FaceMLKit? = null

        /**
         * Get singleton instance of FaceMLKit
         *
         * @param context Application context
         * @return FaceMLKit instance
         */
        fun getInstance(context: Context): FaceMLKit {
            return instance ?: synchronized(this) {
                instance ?: FaceMLKit(context).also { instance = it }
            }
        }
    }

    /**
     * Detect faces in the provided bitmap (callback version)
     *
     * @param bitmap Input image
     * @param callback Callback function that receives the list of detected faces
     */
    fun detectFaces(bitmap: Bitmap, callback: (FaceDetectionResult) -> Unit) {
        coroutineScope.launch {
            val result = detectFaces(bitmap)
            withContext(Dispatchers.Main) {
                callback(result)
            }
        }
    }

    /**
     * Detect faces in the provided bitmap (suspend function version)
     *
     * @param bitmap Input image
     * @return FaceDetectionResult with detected faces
     */
    suspend fun detectFaces(bitmap: Bitmap): FaceDetectionResult {
        val faces = faceDetector.detectFaces(bitmap)
        return FaceDetectionResult(
            success = faces.isNotEmpty(),
            faces = faces,
            message = if (faces.isEmpty()) "No faces detected" else "${faces.size} faces detected"
        )
    }

    /**
     * Extract the largest face from the input image (callback version)
     *
     * @param bitmap Input image
     * @param callback Callback function that receives the processed face bitmap
     */
    fun extractLargestFace(bitmap: Bitmap, callback: (Bitmap?) -> Unit) {
        coroutineScope.launch {
            val result = faceDetector.extractLargestFace(bitmap)
            withContext(Dispatchers.Main) {
                callback(result.faceBitmap)
            }
        }
    }

    /**
     * Extract the largest face from the input image with quality check (callback version)
     *
     * @param bitmap Input image
     * @param callback Callback function that receives the processed face bitmap and quality result
     */
    fun extractLargestFaceWithQuality(bitmap: Bitmap, callback: (Bitmap?, FaceQualityResult) -> Unit) {
        coroutineScope.launch {
            val result = faceDetector.extractLargestFace(bitmap)
            withContext(Dispatchers.Main) {
                callback(result.faceBitmap, result.qualityResult)
            }
        }
    }

    /**
     * Extract the largest face from the input image (suspend function version)
     *
     * @param bitmap Input image
     * @return The processed face bitmap or null if no face detected
     */
    suspend fun extractLargestFace(bitmap: Bitmap): Bitmap? {
        return faceDetector.extractLargestFace(bitmap).faceBitmap
    }

    /**
     * Extract the largest face from the input image with quality assessment (suspend function version)
     *
     * @param bitmap Input image
     * @return Pair containing the processed face bitmap and quality assessment
     */
    suspend fun extractLargestFaceWithQuality(bitmap: Bitmap): Pair<Bitmap?, FaceQualityResult> {
        val result = faceDetector.extractLargestFace(bitmap)
        return Pair(result.faceBitmap, result.qualityResult)
    }

    /**
     * Assess if the face meets quality criteria (callback version)
     *
     * @param bitmap Input image with face
     * @param callback Callback with quality assessment result
     */
    fun assessFaceQuality(bitmap: Bitmap, callback: (FaceQualityResult) -> Unit) {
        coroutineScope.launch {
            val result = assessFaceQuality(bitmap)
            withContext(Dispatchers.Main) {
                callback(result)
            }
        }
    }

    /**
     * Assess if the face meets quality criteria (suspend function version)
     *
     * @param bitmap Input image with face
     * @return Comprehensive quality assessment result
     */
    suspend fun assessFaceQuality(bitmap: Bitmap): FaceQualityResult {
        return faceDetector.assessFaceQuality(bitmap)
    }

    /**
     * Release ML Kit detector resources
     * Call this method when the SDK is no longer needed
     */
    fun close() {
        faceDetector.close()
    }
}