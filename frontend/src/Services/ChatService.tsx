// src/Services/ChatService.ts
import axios from "axios";

axios.defaults.withCredentials = true;

const API_URL = `${import.meta.env.VITE_BACKEND_URL}/chat`;

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

// --- Fetch all chat sessions for a user ---
export async function getChatSessions(userId: string): Promise<ChatSession[]> {
  const response = await axios.get(`${API_URL}/chats`, {
    params: { userId },
  });
  return response.data;
}

// --- Fetch messages for a specific chat session ---
export async function getChatMessages(chatId: string): Promise<Message[]> {
  const response = await axios.get(`${API_URL}/chats/${chatId}/messages`);
  return response.data;
}

// --- Create new chat session ---
export async function createNewChat(userId: string): Promise<ChatSession> {
  const response = await axios.post(`${API_URL}/chats`, { userId });
  return response.data;
}

// --- Send a message (user → bot) ---
export async function sendMessage(
  chatId: string,
  userId: string,
  text: string
): Promise<Message[]> {
  const response = await axios.post(`${API_URL}/chats/${chatId}/send`, {
    userId,
    text,
  });
  return response.data; // returns updated message list or new bot reply
}
