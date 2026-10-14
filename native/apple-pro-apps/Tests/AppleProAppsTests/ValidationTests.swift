import MCP
import Testing

@testable import AppleProApps

private struct ValidationCase: Sendable {
  let value: Value
  let schema: Value
}

struct ValidationTests {
  @Test(arguments: [
    ValidationCase(
      value: .object([:]), schema: .object(["type": .string("object"), "properties": .object([:])])),
    ValidationCase(
      value: .array([]),
      schema: .object(["type": .string("array"), "items": .object(["type": .string("string")])])),
    ValidationCase(value: .string(""), schema: .object(["type": .string("string")])),
    ValidationCase(value: .int(Int.min), schema: .object(["type": .string("integer")])),
    ValidationCase(value: .int(Int.max), schema: .object(["type": .string("integer")])),
  ])
  private func documentedMissingBoundDefaults(_ input: ValidationCase) {
    #expect(throws: Never.self) { try validate(input.value, schema: input.schema) }
  }

  @Test(arguments: [
    Value.null,
    .object([:]),
    .object(["type": .string("unsupported")]),
    .object(["type": .string("object"), "properties": .object([:]), "required": .array([.int(1)])]),
  ])
  func malformedInternalSchemasFailClosed(_ schema: Value) {
    #expect(throws: (any Error).self) { try validate(.object([:]), schema: schema) }
  }
}
