plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.speakerdetect"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.speakerdetect"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "3.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }
}

dependencies {
    // ONNX Runtime — runs on Pixel Tensor TPU/GPU via NNAPI
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.16.3")

    // Android UI
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.core:core-ktx:1.12.0")

    // Coroutines for async processing
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
}
