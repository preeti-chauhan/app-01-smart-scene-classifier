import CoreML
import UIKit
import Accelerate

struct Prediction {
    let label: String
    let confidence: Float  // 0.0 – 1.0
}

class ClassifierService {

    private let model: vit_scene_classifier

    // Same 10 classes used during training (order matters — matches logit index)
    static let classes = ["beach", "forest", "mountain", "kitchen", "bedroom",
                          "street", "restaurant", "office", "living room", "park"]

    // ImageNet normalisation values — same as torchvision Normalize() in notebook 03
    private let mean: [Float] = [0.485, 0.456, 0.406]
    private let std:  [Float] = [0.229, 0.224, 0.225]

    init() {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        guard let m = try? vit_scene_classifier(configuration: config) else {
            fatalError("Failed to load vit_scene_classifier.mlpackage")
        }
        self.model = m
    }

    /// Classify a UIImage and return top-3 predictions.
    func classify(image: UIImage, completion: @escaping ([Prediction]) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            guard let input = self.preprocess(image: image),
                  let output = try? self.model.prediction(image: input) else {
                DispatchQueue.main.async { completion([]) }
                return
            }

            // output.logits is MLMultiArray shape [1, 10]
            let logits = (0..<10).map { output.logits[[0, $0] as [NSNumber]].floatValue }
            let probs  = self.softmax(logits)

            let top3 = probs.enumerated()
                .sorted { $0.element > $1.element }
                .prefix(3)
                .map { Prediction(label: Self.classes[$0.offset], confidence: $0.element) }

            DispatchQueue.main.async { completion(Array(top3)) }
        }
    }

    // MARK: - Private helpers

    /// Resize to 224×224, convert to float, apply ImageNet normalisation.
    /// Equivalent to: transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean, std)
    private func preprocess(image: UIImage) -> MLMultiArray? {
        let size = CGSize(width: 224, height: 224)

        // Resize + center-crop to 224×224
        UIGraphicsBeginImageContextWithOptions(size, true, 1.0)
        image.draw(in: CGRect(origin: .zero, size: size))
        let resized = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        guard let cgImage = resized?.cgImage else { return nil }

        let width = 224, height = 224
        var pixelData = [UInt8](repeating: 0, count: width * height * 4)  // RGBA

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: &pixelData,
                                      width: width, height: height,
                                      bitsPerComponent: 8, bytesPerRow: width * 4,
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Shape: [1, 3, 224, 224] — same layout as PyTorch CHW tensors
        guard let array = try? MLMultiArray(shape: [1, 3, 224, 224], dataType: .float32) else { return nil }

        for y in 0..<height {
            for x in 0..<width {
                let pixelIndex = (y * width + x) * 4
                for c in 0..<3 {
                    let raw = Float(pixelData[pixelIndex + c]) / 255.0
                    let normalised = (raw - mean[c]) / std[c]
                    // CHW index: channel * H * W + row * W + col
                    array[[0, c, y, x] as [NSNumber]] = NSNumber(value: normalised)
                }
            }
        }
        return array
    }

    /// Softmax: converts raw logits to probabilities that sum to 1.
    /// Same as F.softmax(logits, dim=-1) in PyTorch.
    private func softmax(_ logits: [Float]) -> [Float] {
        let maxVal = logits.max() ?? 0
        let exps = logits.map { exp($0 - maxVal) }
        let sum  = exps.reduce(0, +)
        return exps.map { $0 / sum }
    }
}
