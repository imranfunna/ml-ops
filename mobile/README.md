# FlowSure Edge — On-Device (Android / iOS)

Dezelfde `model.onnx` uit `/Volumes/flowsure/mlops/artifacts/models/edge_onnx/`
draait via **ONNX Runtime Mobile** on-device. Model is < 3 MB, inferentie ~1-5 ms.

Voordelen:
- Geen netwerk-round-trip → snellere UI-suggesties
- Werkt offline
- Ticket-tekst verlaat het toestel niet (GDPR / dataminimalisatie)

## Android (Kotlin)

**`app/build.gradle.kts`**
```kotlin
dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")
}
```

Kopieer `model.onnx` + `labels.json` naar `app/src/main/assets/`.

**`FlowSureEdge.kt`**
```kotlin
import ai.onnxruntime.*
import android.content.Context
import org.json.JSONArray

class FlowSureEdge(ctx: Context) {
    private val env = OrtEnvironment.getEnvironment()
    private val session = env.createSession(
        ctx.assets.open("model.onnx").readBytes()
    )
    private val labels: List<String> =
        JSONArray(ctx.assets.open("labels.json").bufferedReader().readText())
            .let { arr -> List(arr.length()) { arr.getString(it) } }

    data class Result(val category: String, val confidence: Float)

    fun predict(text: String): Result {
        val input = OnnxTensor.createTensor(env, arrayOf(arrayOf(text)))
        session.run(mapOf("input_text" to input)).use { out ->
            val pred = (out[0].value as LongArray)[0].toInt()
            val prob = (out[1].value as Array<FloatArray>)[0]
            return Result(labels[pred], prob.max())
        }
    }
}
```

## iOS (Swift)

**`Podfile`**
```ruby
pod 'onnxruntime-objc', '~> 1.19.0'
```

Sleep `model.onnx` + `labels.json` in je Xcode-project (Copy Bundle Resources).

**`FlowSureEdge.swift`**
```swift
import Foundation
import onnxruntime_objc

final class FlowSureEdge {
    private let session: ORTSession
    private let labels: [String]

    init() throws {
        let env = try ORTEnv(loggingLevel: .warning)
        let path = Bundle.main.path(forResource: "model", ofType: "onnx")!
        session = try ORTSession(env: env, modelPath: path, sessionOptions: nil)

        let lblURL = Bundle.main.url(forResource: "labels", withExtension: "json")!
        labels = try JSONDecoder().decode([String].self, from: Data(contentsOf: lblURL))
    }

    func predict(_ text: String) throws -> (category: String, confidence: Float) {
        let input = try ORTValue(
            tensorStringData: [text],
            shape: [1, 1] as [NSNumber])
        let out = try session.run(
            withInputs: ["input_text": input],
            outputNames: ["label", "probabilities"],
            runOptions: nil)
        let pred = try (out["label"]!.tensorData() as NSData)
            .bindMemory(to: Int64.self, capacity: 1).pointee
        let probs = try (out["probabilities"]!.tensorData() as NSData)
        let ptr = probs.bytes.bindMemory(to: Float.self, capacity: labels.count)
        let confidence = (0..<labels.count).map { ptr[$0] }.max() ?? 0
        return (labels[Int(pred)], confidence)
    }
}
```

## Cross-platform (React Native / Flutter)

| Framework      | Package                                                         |
| -------------- | --------------------------------------------------------------- |
| React Native   | `onnxruntime-react-native`                                      |
| Flutter        | `onnxruntime` (pub.dev)                                         |
| Expo           | Werkt niet — heeft dev-client of bare workflow nodig            |

## Model updaten op geproduceerde toestellen

Voor productie **niet** hard bundelen in de APK/IPA — host `model.onnx` +
`labels.json` extern zodat je zonder store-review kunt updaten.

Waar de CI ze publiceert (`.github/workflows/ci-cd.yml`, job `docker_publish`):

1. **UC Volume** — canonical source, `/Volumes/flowsure/mlops/artifacts/mobile/`
2. **GitHub Release** — bij elke `v*` tag als downloadbare assets
3. **GHCR container** — `ghcr.io/<org>/flowsure-edge:<tag>` bevat dezelfde files

Je mobile-app kan bij startup checken tegen de GitHub Releases API (of een
eigen CDN als je die hebt) en het model on-demand ophalen met een versie-check
+ ETag-caching.
