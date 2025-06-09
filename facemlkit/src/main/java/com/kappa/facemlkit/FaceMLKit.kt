package com.kappa.facemlkit

import android.content.Context
import android.graphics.Bitmap
import com.kappa.facemlkit.detector.FaceDetector
import com.kappa.facemlkit.models.FaceDetectionResult
import com.kappa.facemlkit.models.FaceQualityResult
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume

/**
 * FaceMLKit - Main entry point for the face detection and processing SDK
 */
class FaceMLKit private constructor(context: Context) {

    private val faceDetector = FaceDetector()
    private val coroutineScope = CoroutineScope(Dispatchers.Default)

    companion object {
        @Volatile private var instance: FaceMLKit? = null

        /**
         * Get singleton instance of FaceMLKit
         */
        fun getInstance(context: Context): FaceMLKit {
            return instance ?: synchronized(this) {
                instance ?: FaceMLKit(context).also { instance = it }
            }
        }
    }

    /**
     * Detect faces in the provided bitmap (callback)
     */
    fun detectFaces(bitmap: Bitmap, callback: (FaceDetectionResult) -> Unit) {
        coroutineScope.launch {
            val result = faceDetector.detectFaces(bitmap)
            withContext(Dispatchers.Main) {
                callback(
                    FaceDetectionResult(
                        success = result.isNotEmpty(),
                        faces = result,
                        message = if (result.isEmpty()) "No faces detected" else "${result.size} faces detected"
                    )
                )
            }
        }
    }

    /**
     * Extract the largest face from the input image (callback)
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
     * Extract the largest face from the input image with quality check (callback)
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
     * Suspend version of detectFaces, used internally by coroutine-based consumers
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
     * Suspend version of extractLargestFace, used internally by coroutine-based consumers
     */
    suspend fun extractLargestFace(bitmap: Bitmap): Bitmap? {
        return faceDetector.extractLargestFace(bitmap).faceBitmap
    }

    /**
     * Suspend version of extractLargestFaceWithQuality
     */
    suspend fun extractLargestFaceWithQuality(bitmap: Bitmap): Pair<Bitmap?, FaceQualityResult> =
        suspendCancellableCoroutine { continuation ->
            extractLargestFaceWithQuality(bitmap) { faceBitmap, qualityResult ->
                continuation.resume(Pair(faceBitmap, qualityResult))
            }
        }

    /**
     * Release ML Kit detector resources
     */
    fun close() {
        faceDetector.close()
    }
}