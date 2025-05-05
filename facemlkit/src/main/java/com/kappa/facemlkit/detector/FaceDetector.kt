package com.kappa.facemlkit.detector

import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
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

    private val faceDetectorOptions = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
        .setMinFaceSize(0.15f)
        .enableTracking()
        .build()

    private val detector = FaceDetection.getClient(faceDetectorOptions)

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
     * @return Processed face bitmap or null if no face detected
     */
    suspend fun extractLargestFace(bitmap: Bitmap): Bitmap? = withContext(Dispatchers.IO) {
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
                // Instead of returning null, apply the resize and crop
                return@withContext ImageUtils.applyResizeAndCenterCrop(bitmap)
            }

            val largestFace = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }
                ?: return@withContext ImageUtils.applyResizeAndCenterCrop(bitmap)

            val box = largestFace.boundingBox
            Log.d(TAG, "Selected face bounding box: x=${box.left}, y=${box.top}, w=${box.width()}, h=${box.height()}")

            // Crop tightly to the bounding box
            val faceBitmap = ImageUtils.cropFaceTightly(bitmap, box)

            if (faceBitmap != null) {
                Log.d(TAG, "Face cropped successfully, size: ${faceBitmap.width}x${faceBitmap.height}")
                // Apply resize and center crop to the face bitmap
                return@withContext ImageUtils.applyResizeAndCenterCrop(faceBitmap)
            } else {
                Log.e(TAG, "Invalid crop dimensions: $box, using fallback")
                return@withContext ImageUtils.applyResizeAndCenterCrop(bitmap)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error extracting largest face: ${e.message}", e)
            // On error, apply the fallback approach rather than returning null
            try {
                return@withContext ImageUtils.applyResizeAndCenterCrop(bitmap)
            } catch (e2: Exception) {
                Log.e(TAG, "Fallback also failed: ${e2.message}", e2)
                return@withContext null
            }
        }
    }

    /**
     * Release ML Kit detector
     */
    fun close() {
        detector.close()
    }
}