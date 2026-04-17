//
//  ContentView.swift
//  SmartSceneClassifier
//
//  Created by Preeti Chauhan on 4/8/26.
//

import SwiftUI
import PhotosUI

struct ContentView: View {

    @State private var selectedPhoto: PhotosPickerItem?
    @State private var displayImage: Image?
    @State private var uiImage: UIImage?
    @State private var predictions: [Prediction] = []
    @State private var isClassifying = false

    private let classifier = ClassifierService()

    var body: some View {
        NavigationStack {
            GeometryReader { geo in
              let w = geo.size.width - 32  // account for horizontal padding
              ScrollView {
                VStack(spacing: 20) {

                // ── Photo picker ──────────────────────────────────────────
                PhotosPicker(selection: $selectedPhoto, matching: .images) {
                    if let displayImage {
                        displayImage
                            .resizable()
                            .scaledToFill()
                            .frame(width: w, height: 280)
                            .clipped()
                            .cornerRadius(16)
                            .overlay(alignment: .bottomTrailing) {
                                Label("Change", systemImage: "photo.badge.plus")
                                    .font(.caption)
                                    .fontWeight(.semibold)
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 6)
                                    .background(.ultraThinMaterial)
                                    .cornerRadius(20)
                                    .padding(10)
                            }
                    } else {
                        RoundedRectangle(cornerRadius: 16)
                            .fill(Color(.systemGray6))
                            .frame(width: w, height: 280)
                            .overlay {
                                VStack(spacing: 8) {
                                    Image(systemName: "photo.badge.plus")
                                        .font(.system(size: 44))
                                        .foregroundStyle(.secondary)
                                    Text("Tap to choose a photo")
                                        .foregroundStyle(.secondary)
                                }
                            }
                    }
                }
                .onChange(of: selectedPhoto) { _, newItem in
                    Task {
                        if let data = try? await newItem?.loadTransferable(type: Data.self),
                           let ui = UIImage(data: data) {
                            uiImage = ui
                            displayImage = Image(uiImage: ui)
                            predictions = []
                            classifyImage(ui)
                        }
                    }
                }

                // ── Classify button ───────────────────────────────────────
                if let image = uiImage {
                    Button {
                        classifyImage(image)
                    } label: {
                        HStack {
                            if isClassifying {
                                ProgressView().tint(.white).padding(.trailing, 4)
                            }
                            Text(isClassifying ? "Classifying…" : "Classify Scene")
                                .fontWeight(.semibold)
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.accentColor)
                        .foregroundStyle(.white)
                        .cornerRadius(12)
                    }
                    .disabled(isClassifying)
                }

                // ── Results ───────────────────────────────────────────────
                if !predictions.isEmpty {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Scene Classification")
                            .font(.headline)
                        ForEach(Array(predictions.enumerated()), id: \.offset) { index, pred in
                            PredictionRow(prediction: pred, rank: index + 1)
                        }

                        Divider()

                        // Low confidence warning
                        if let top = predictions.first, top.confidence < 0.4 {
                            Label("Low confidence — image may not match any supported scene.", systemImage: "exclamationmark.triangle.fill")
                                .font(.caption)
                                .foregroundStyle(.orange)
                        }

                        // Scene-only note
                        Label("Classifies scenes, not people or objects. Point at a place.", systemImage: "info.circle")
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        // Supported classes
                        Text("Supported: beach · forest · mountain · kitchen · bedroom · street · restaurant · office · living room · park")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(16)
                }

                    Spacer()
                }
                .padding(.horizontal, 16)
                .padding(.top, 8)
              }
            }
            .navigationTitle("Smart Scene Classifier")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private func classifyImage(_ image: UIImage) {
        isClassifying = true
        classifier.classify(image: image) { results in
            self.predictions = results
            self.isClassifying = false
        }
    }
}

// ── Prediction row ────────────────────────────────────────────────────────────
struct PredictionRow: View {
    let prediction: Prediction
    let rank: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("\(rank). \(prediction.label.capitalized)")
                    .fontWeight(rank == 1 ? .bold : .regular)
                Spacer()
                Text(String(format: "%.1f%%", prediction.confidence * 100))
                    .foregroundStyle(rank == 1 ? Color.accentColor : .secondary)
                    .fontWeight(rank == 1 ? .bold : .regular)
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color(.systemGray5))
                        .frame(height: 6)
                    RoundedRectangle(cornerRadius: 4)
                        .fill(rank == 1 ? Color.accentColor : Color.secondary)
                        .frame(width: geo.size.width * CGFloat(prediction.confidence), height: 6)
                }
            }
            .frame(height: 6)
        }
    }
}

#Preview {
    ContentView()
}
