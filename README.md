# FaceMLKit

A powerful Android SDK for face detection and quality assessment using Google ML Kit. FaceMLKit provides easy-to-use APIs for detecting faces, extracting face regions, and assessing face quality with advanced image processing techniques.

## Features

- **Face Detection**: Accurate face detection using Google ML Kit
- **Face Extraction**: Extract and crop the largest face from images
- **Quality Assessment**: Advanced face quality checking with blur detection
- **Image Processing**: Automatic brightness adjustment and image enhancement
- **Fallback Processing**: Robust handling when no faces are detected
- **Coroutine Support**: Both callback-based and suspend function APIs

## Installation

Add the following dependency to your `build.gradle` file:

```gradle
dependencies {
    implementation(files("libs/facemlkit-release.aar"))
    
    // Required ML Kit dependencies
    implementation 'com.google.mlkit:face-detection:16.1.5'
}
```

## API Reference

### FaceMLKit Class

#### getInstance(context: Context): FaceMLKit
Get singleton instance of FaceMLKit.

#### detectFaces(bitmap: Bitmap, callback: (FaceDetectionResult) -> Unit)
Detect all faces in the provided bitmap and return results via callback.

#### detectFaces(bitmap: Bitmap): FaceDetectionResult
Suspend version of face detection that returns all detected faces.

#### extractLargestFace(bitmap: Bitmap, callback: (Bitmap?) -> Unit)
Find the largest face in the image, crop it, resize to 224x224, and return via callback.

#### extractLargestFace(bitmap: Bitmap): Bitmap?
Suspend version that extracts and returns the largest face as a 224x224 bitmap.

#### extractLargestFaceWithQuality(bitmap: Bitmap, callback: (Bitmap?, FaceQualityResult) -> Unit)
Extract the largest face and perform quality assessment, returning both the face bitmap and quality results via callback.

#### extractLargestFaceWithQuality(bitmap: Bitmap): Pair<Bitmap?, FaceQualityResult>
Suspend version that extracts the largest face and returns both the bitmap and quality assessment.

#### close()
Release ML Kit detector resources and clean up.

## Data Models

### FaceDetectionResult
```kotlin
data class FaceDetectionResult(
    val success: Boolean,           // Whether detection found any faces
    val faces: List<Face>,          // List of ML Kit Face objects
    val message: String             // Status message about detection result
)
```

### FaceQualityResult
```kotlin
data class FaceQualityResult(
    val isGoodQuality: Boolean,     // Whether face passes quality checks
    val qualityScore: Float,        // Quality score from 0.0 to 1.0
    val laplacianRawScore: Double,  // Raw sharpness measurement
    val issues: List<QualityIssue>, // List of detected problems
    val failureReason: String?      // Description of quality failure
)
```

### QualityIssue
```kotlin
enum class QualityIssue {
    MULTIPLE_FACES,      // More than one face detected
    NO_FACE_DETECTED,    // No face found in image
    UNDEREXPOSED,        // Image is too dark
    BLURRY_FACE         // Face lacks sharpness
}
```

## Usage Examples

### Basic Face Detection
```kotlin
val faceMLKit = FaceMLKit.getInstance(context)

// Callback version
faceMLKit.detectFaces(bitmap) { result ->
    if (result.success) {
        Log.d("Faces", "Found ${result.faces.size} faces")
    }
}

// Coroutine version
val result = faceMLKit.detectFaces(bitmap)
```

### Face Extraction
```kotlin
// Extract face only
faceMLKit.extractLargestFace(bitmap) { faceBitmap ->
    faceBitmap?.let { 
        // Use 224x224 face bitmap
    }
}

// Extract with quality check
faceMLKit.extractLargestFaceWithQuality(bitmap) { faceBitmap, quality ->
    if (quality.isGoodQuality) {
        // Use high quality face
    }
}
```

## Requirements

- Android API level 21+
- Google ML Kit Face Detection
- Kotlin Coroutines support