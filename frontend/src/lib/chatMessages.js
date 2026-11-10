// The chat panel's message list as plain data: what goes back to the API
// as history, and how the answer being written changes as stream events
// arrive (api.streamChatMessage). No React here (chatMessages.test.js).
//
// A message is { role, content } plus, for the assistant's answer,
// { toolCalls, pending, activeTool, stopReason }. `pending` marks the
// answer still being streamed; it is always the last message.

// Only finished turns are history: an answer that failed midway is not
// something the model said.
export function historyFor(messages) {
  return messages.filter((m) => !m.pending && m.content).map((m) => ({ role: m.role, content: m.content }));
}

// The list right after the user sends `text`: their message, and an
// empty answer waiting for the stream.
export function withQuestion(messages, text) {
  return [...messages, { role: "user", content: text }, { role: "assistant", content: "", toolCalls: [], pending: true }];
}

function updateLast(messages, fn) {
  return messages.map((m, i) => (i === messages.length - 1 ? fn(m) : m));
}

// One stream event applied to the answer being written. A lookup stays
// shown as active until the model writes again: the wait after it is the
// model reading the result, and the blank line the backend sends between
// the two is not "writing".
export function applyStreamEvent(messages, event) {
  if (event.type === "text") {
    return updateLast(messages, (m) => ({ ...m, content: m.content + event.delta, activeTool: event.delta.trim() ? null : m.activeTool }));
  }
  if (event.type === "tool") {
    return updateLast(messages, (m) => ({ ...m, toolCalls: [...m.toolCalls, { name: event.name, input: event.input }], activeTool: event.name }));
  }
  return messages;
}

// The stream ended normally: the "done" event carries the whole answer.
export function withAnswer(messages, done) {
  return updateLast(messages, (m) => ({ ...m, content: done.reply, toolCalls: done.tool_calls, pending: false, activeTool: null, stopReason: done.stop_reason }));
}

// The stream failed: keep what was written, drop an answer that never started.
export function afterFailure(messages) {
  return messages.filter((m) => !(m.pending && !m.content)).map((m) => (m.pending ? { ...m, pending: false, activeTool: null } : m));
}
