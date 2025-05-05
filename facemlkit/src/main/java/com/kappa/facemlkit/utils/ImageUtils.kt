package com.kappa.facemlkit.utils

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import android.util.Log
import java.io.ByteArrayOutputStream

/**
 * Utility functions for image processing
 * Internal class not exposed directly to SDK users
 */
internal object ImageUtils {

    private const val TAG = "ImageUtils"

    /**
     * Apply the resize and center crop transformations
     * Matches Python transforms.Resize(256) followed by transforms.CenterCrop(224)
     *
     * @param bitmap Input bitmap
     * @return Processed bitmap with standard size (224x224)
     */
    fun applyResizeAndCenterCrop(bitmap: Bitmap): Bitmap {
        // Step 1: Resize to 256x256
        val resized = Bitmap.createScaledBitmap(bitmap, 256, 256, true)
        Log.d(TAG, "Resized image size: ${resized.width}x${resized.height}")

        // Step 2: Center crop to 224x224
        val x = (resized.width - 224) / 2
        val y = (resized.height - 224) / 2
        val cropped = Bitmap.createBitmap(resized, x, y, 224, 224)
        Log.d(TAG, "Cropped image size: ${cropped.width}x${cropped.height}")

        return cropped
    }

    /**
     * Crop exactly to the ML Kit bounding box (zero margin)
     *
     * @param bitmap Input bitmap
     * @param boundingBox Face bounding box
     * @return Cropped face or null if invalid dimensions
     */
    fun cropFaceTightly(bitmap: Bitmap, boundingBox: Rect): Bitmap? {
        // Clamp to image bounds
        val left = boundingBox.left.coerceAtLeast(0)
        val top = boundingBox.top.coerceAtLeast(0)
        val right = boundingBox.right.coerceAtMost(bitmap.width)
        val bottom = boundingBox.bottom.coerceAtMost(bitmap.height)

        val width = right - left
        val height = bottom - top

        Log.d(TAG, "Tight crop box: left=$left top=$top width=$width height=$height")

        if (width <= 0 || height <= 0) return null
        return Bitmap.createBitmap(bitmap, left, top, width, height)
    }

    /**
     * Compress bitmap for fallback detection
     *
     * @param bitmap Input bitmap
     * @param quality Compression quality (0-100)
     * @return Compressed bitmap
     */
    fun compressBitmap(bitmap: Bitmap, quality: Int): Bitmap {
        val outputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, quality, outputStream)
        val byteArray = outputStream.toByteArray()
        return BitmapFactory.decodeByteArray(byteArray, 0, byteArray.size)
    }

    /**
     * Create a bitmap with expanded margins around a face
     *
     * @param bitmap Original image
     * @param boundingBox Face bounding box
     * @param marginPercent Margin to add around face (percentage of face dimensions)
     * @return Bitmap with margins around face
     */
    fun cropFaceWithMargin(bitmap: Bitmap, boundingBox: Rect, marginPercent: Float = 0.3f): Bitmap? {
        val width = boundingBox.width()
        val height = boundingBox.height()

        val widthMargin = (width * marginPercent).toInt()
        val heightMargin = (height * marginPercent).toInt()

        // Calculate new bounds with margins
        val left = (boundingBox.left - widthMargin).coerceAtLeast(0)
        val top = (boundingBox.top - heightMargin).coerceAtLeast(0)
        val right = (boundingBox.right + widthMargin).coerceAtMost(bitmap.width)
        val bottom = (boundingBox.bottom + heightMargin).coerceAtMost(bitmap.height)

        val croppedWidth = right - left
        val croppedHeight = bottom - top

        if (croppedWidth <= 0 || croppedHeight <= 0) return null

        return Bitmap.createBitmap(bitmap, left, top, croppedWidth, croppedHeight)
    }
}