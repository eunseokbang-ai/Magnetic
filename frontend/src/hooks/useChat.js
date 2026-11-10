import { useState } from "react";
import * as api from "../api";
import { afterFailure, applyStreamEvent, historyFor, withAnswer, withQuestion } from "../lib/chatMessages";

// State of the AI interpretation assistant's panel: the message list and
// one streamed turn at a time. Moved out of App.jsx in 2026-11; what a
// stream event does to the list is in lib/chatMessages.js.
export default function useChat({ ensureProject }) {
  const [chatMessages, setChatMessages] = useState([]);
  const [chatSending, setChatSending] = useState(false);
  const [chatError, setChatError] = useState(null);

  const handleSendChatMessage = async (text) => {
    try {
      setChatError(null);
      setChatSending(true);
      const id = await ensureProject();
      const history = historyFor(chatMessages);
      setChatMessages((prev) => withQuestion(prev, text));
      const done = await api.streamChatMessage(id, text, history, (event) => setChatMessages((prev) => applyStreamEvent(prev, event)));
      setChatMessages((prev) => withAnswer(prev, done));
    } catch (e) {
      setChatError(e.message || String(e));
      setChatMessages(afterFailure);
    } finally {
      setChatSending(false);
    }
  };

  return { chatMessages, chatSending, chatError, handleSendChatMessage };
}
