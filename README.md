# 📷💥 Android SDKs Documentation

## Installation

Add to your app's `build.gradle`:

```gradle
dependencies {
    implementation(files("libs/facemlkit-release.aar"))
    implementation(files("libs/imageconverter-release.aar"))
}
```

## 👱🏼 FaceMLKit

A simplified Android SDK for face detection and processing using ML Kit.

### Overview

FaceMLKit provides an easy-to-use interface for:
- Detecting faces in images
- Extracting the largest face
- Assessing face quality

### Usage

```kotlin
// Initialize
val faceMLKit = FaceMLKit.getInstance(context)

// Detect faces
faceMLKit.detectFaces(bitmap) { result ->
    if (result.success) {
        println("Found ${result.faces.size} faces")
    }
}

// Extract largest face
faceMLKit.extractLargestFace(bitmap) { faceBitmap ->
    // Use the extracted face
}

// Assess quality
faceMLKit.assessFaceQuality(bitmap) { isHighQuality ->
    if (isHighQuality) {
        println("Good quality face")
    }
}

// Clean up
faceMLKit.close()
```

### Features

✅ Face detection with bounding boxes  
✅ Automatic face extraction  
✅ Quality assessment based on eye openness  
✅ Automatic 224x224 standard cropping  
✅ Coroutine support  

---

## 🖼️ ImageConverter

Android SDK for efficient YUV to RGB image conversion using RenderScript.

### Overview

ImageConverter efficiently converts CameraX ImageProxy (YUV format) to Android Bitmap (RGB) with GPU acceleration.

### Usage

```kotlin
// Initialize
val imageConverter = ImageConverter.getInstance(context)

// Convert in CameraX analyzer
class ImageAnalyzer : ImageAnalysis.Analyzer {
    override fun analyze(imageProxy: ImageProxy) {
        val bitmap = imageConverter.convertToBitmap(imageProxy)
        
        bitmap?.let {
            // Use the bitmap
        }
        
        imageProxy.close()
    }
}
```

### Features

✅ YUV to RGB conversion  
✅ RenderScript GPU acceleration  
✅ Automatic image rotation  
✅ Thread-safe singleton pattern  
✅ Memory efficient processing  

---

## License

MIT License
