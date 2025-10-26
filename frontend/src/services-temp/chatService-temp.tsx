// src/Services/ChatService.ts
import axios from "axios";

// Enable sending cookies with requests
axios.defaults.withCredentials = true;

// Base API URL for chat-related endpoints
const API_URL = `${import.meta.env.VITE_BACKEND_URL}/chat`;

// Chat session type definition
export interface ChatSession {
  id: string;
  title: string;
  lastMessage?: string;
}

// Message type definition
export interface Message {
  id: string;
  sender: "user" | "bot";
  text: string;
  timestamp: string;
}

// Fetch all chat sessions for a user
export async function getChatSessions(userId: string): Promise<ChatSession[]> {
  const response = await axios.get(`${API_URL}/chats`, {
    params: { userId },
  });
  return response.data;
}

// Fetch messages for a specific chat session
export async function getChatMessages(chatId: string): Promise<Message[]> {
  const response = await axios.get(`${API_URL}/chats/${chatId}/messages`);
  return response.data;
}

// Create new chat session
export async function createNewChat(userId: string): Promise<ChatSession> {
  const response = await axios.post(`${API_URL}/chats`, { userId });
  return response.data;
}

// Send a message from user to bot in a specific chat session
export async function sendMessage(
  chatId: string,
  userId: string,
  text: string
): Promise<Message[]> {
  const response = await axios.post(`${API_URL}/chats/${chatId}/send`, {
    userId,
    text,
  });
  return response.data;
}
