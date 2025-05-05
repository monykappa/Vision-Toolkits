package com.kappa.imageconverter


import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.renderscript.Allocation
import android.renderscript.Element
import android.renderscript.RenderScript
import android.renderscript.ScriptIntrinsicYuvToRGB
import android.util.Log
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy

class ImageConverter private constructor(private val context: Context) {

    companion object {
        @Volatile private var instance: ImageConverter? = null

        fun getInstance(context: Context): ImageConverter {
            return instance ?: synchronized(this) {
                instance ?: ImageConverter(context.applicationContext).also { instance = it }
            }
        }
    }

    @androidx.annotation.OptIn(ExperimentalGetImage::class)
    @OptIn(ExperimentalGetImage::class)
    fun convertToBitmap(imageProxy: ImageProxy): Bitmap? {
        val image = imageProxy.image ?: return null
        val planes = image.planes
        val width = image.width
        val height = image.height

        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]
        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer
        val yRowStride = yPlane.rowStride
        val uRowStride = uPlane.rowStride
        val vRowStride = vPlane.rowStride
        val uPixelStride = uPlane.pixelStride
        val vPixelStride = vPlane.pixelStride

        Log.d("ImageConverter", "Y row stride: $yRowStride, U row stride: $uRowStride, V row stride: $vRowStride")
        Log.d("ImageConverter", "U pixel stride: $uPixelStride, V pixel stride: $vPixelStride")

        val ySize = width * height
        val uvWidth = width / 2
        val uvHeight = height / 2
        val totalSize = ySize + uvWidth * uvHeight * 2
        val yuvData = ByteArray(totalSize)

        for (row in 0 until height) {
            val offset = row * yRowStride
            yBuffer.position(offset)
            val length = minOf(width, yBuffer.remaining())
            yBuffer.get(yuvData, row * width, length)
            for (i in length until width) {
                yuvData[row * width + i] = 0
            }
        }

        val uvOffset = ySize
        for (row in 0 until uvHeight) {
            for (col in 0 until uvWidth) {
                val idx = row * uvWidth + col
                val uPos = row * uRowStride + col * uPixelStride
                val vPos = row * vRowStride + col * vPixelStride
                val uByte = if (uPos < uBuffer.remaining()) uBuffer.get(uPos) else 128.toByte()
                val vByte = if (vPos < vBuffer.remaining()) vBuffer.get(vPos) else 128.toByte()
                yuvData[uvOffset + idx * 2] = vByte
                yuvData[uvOffset + idx * 2 + 1] = uByte
            }
        }

        val rs = RenderScript.create(context)
        val yuvToRgb = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs))
        val inputAlloc = Allocation.createSized(rs, Element.U8(rs), totalSize)
        inputAlloc.copyFrom(yuvData)

        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val outputAlloc = Allocation.createFromBitmap(rs, bitmap)

        yuvToRgb.setInput(inputAlloc)
        yuvToRgb.forEach(outputAlloc)
        outputAlloc.copyTo(bitmap)

        inputAlloc.destroy()
        outputAlloc.destroy()
        yuvToRgb.destroy()
        rs.destroy()

        val rotation = imageProxy.imageInfo.rotationDegrees
        return if (rotation != 0) {
            val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } else {
            bitmap
        }
    }
}
