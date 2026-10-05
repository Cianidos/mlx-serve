import XCTest
import AppKit
@testable import MLXCore

final class GenerationDefaultsRowsTests: XCTestCase {
    func testInheritanceUmbrellaTracksOffOnMixedAndPreservesExistingValues() {
        var profile = GenerationDefaults()
        XCTAssertEqual(profile.inheritanceState(), true)
        profile.setInherited(false, field: .temperature)
        XCTAssertNil(profile.inheritanceState())
        profile.rules["temperature"] = .init(value: .number(0.25), ignoreClient: true)
        profile.setAllInherited(false)
        XCTAssertEqual(profile.inheritanceState(), false)
        XCTAssertEqual(profile.rules["temperature"]?.value, .number(0.25))
        XCTAssertTrue(profile.rules["temperature"]!.ignoreClient)
        profile.setAllInherited(true)
        XCTAssertEqual(profile.inheritanceState(), true)
        XCTAssertTrue(profile.rules.isEmpty)
    }

    @MainActor
    func testNativeUmbrellasRenderMixedStateAndClickSelectsAll() {
        var clicked: Bool?
        let button = MixedCheckboxButton(title: "Default", value: nil) { clicked = $0 }
        XCTAssertEqual(button.state, .mixed)
        XCTAssertTrue(button.allowsMixedState)
        button.performClick(nil)
        XCTAssertEqual(clicked, true)
        button.setValue(true)
        XCTAssertEqual(button.state, .on)
        button.performClick(nil)
        XCTAssertEqual(clicked, false)
        button.setValue(false)
        XCTAssertEqual(button.state, .off)
    }

    func testLockUmbrellaTouchesOnlyConfiguredRowsAndTracksMixedState() {
        var profile = GenerationDefaults()
        XCTAssertEqual(profile.clientLockState(), false)
        profile.rules["temperature"] = .init(value: .number(0.5), ignoreClient: true)
        profile.rules["top_k"] = .init(value: .number(0))
        XCTAssertNil(profile.clientLockState())
        profile.setAllClientLocks(true)
        XCTAssertEqual(profile.clientLockState(), true)
        XCTAssertEqual(profile.rules.count, 2)
        XCTAssertEqual(profile.rules["top_k"]?.value, .number(0))
        profile.setAllClientLocks(false)
        XCTAssertEqual(profile.clientLockState(), false)
        XCTAssertNil(profile.rules["reasoning_budget"])
    }

    func testInheritedClientBodyDoesNotPinSamplingAndRemoteBodyKeepsExistingDefaults() throws {
        var local = APIClient.RequestDefaults()
        local.inheritGeneration = true
        let data = try APIClient.chatRequestBody(messages: [], maxTokens: 64, temperature: 0.8,
                                                 enableThinking: false, defaults: local)
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertNil(body["temperature"])
        XCTAssertNil(body["top_p"])
        XCTAssertNil(body["max_tokens"])
        XCTAssertEqual(body["enable_thinking"] as? Bool, false)
        let remote = try APIClient.chatRequestBody(messages: [], maxTokens: 64, temperature: 0.8, enableThinking: false)
        let remoteBody = try XCTUnwrap(JSONSerialization.jsonObject(with: remote) as? [String: Any])
        XCTAssertEqual(remoteBody["temperature"] as? Double, 0.8)
        XCTAssertEqual(remoteBody["top_p"] as? Double, 0.95)
        XCTAssertEqual(remoteBody["max_tokens"] as? Int, 64)
        XCTAssertNil(remoteBody["enable_thinking"])
    }
}
