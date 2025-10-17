// src/Services/ChatService.ts
import axios from "axios";

const API_URL = `${import.meta.env.VITE_API_URL}/chat`;

export interface ChatSession {
  id: string;
  title: string;
  lastMessage?: string;
}

export interface Message {
  id: string;
  sender: "user" | "bot";
  text: string;
  timestamp: string;
}

// --- Fetch all chat sessions for current user ---
export async function getChatSessions(userId: string): Promise<ChatSession[]> {
  const response = await axios.get(`${API_URL}/chats`, {
    params: { userId },
    withCredentials: true,
  });
  return response.data;
}

// --- Fetch messages for a specific chat session ---
export async function getChatMessages(chatId: string): Promise<Message[]> {
  const response = await axios.get(`${API_URL}/chats/${chatId}/messages`, {
    withCredentials: true,
  });
  return response.data;
}

// --- Create new chat session ---
export async function createNewChat(userId: string): Promise<ChatSession> {
  const response = await axios.post(
    `${API_URL}/chats`,
    { userId },
    { withCredentials: true }
  );

  return response.data;
}

// --- Send a message (user → bot) ---
export async function sendMessage(
  chatId: string,
  userId: string,
  text: string
): Promise<Message[]> {
  const response = await axios.post(
    `${API_URL}/chats/${chatId}/send`,
    { userId, text },
    { withCredentials: true }
  );
  return response.data; // returns updated message list or new bot reply
}
