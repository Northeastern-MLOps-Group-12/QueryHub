// src/Services/ChatService.ts
import axios from "axios";

const API_URL = `${import.meta.env.VITE_API_URL}/chat`;

export interface ChatSession {
  id: string;
  title: string;
  createdAt: string;
  lastMessage?: string;
}

export interface Message {
  id: string;
  sender: "user" | "bot";
  text: string;
  createdAt: string;
  chatId: string;
}

// Get all chat sessions for the current user
export const getChatSessions = async (): Promise<ChatSession[]> => {
  const res = await axios.get(`${API_URL}/sessions`, { withCredentials: true });
  return res.data;
};

// Get messages of a specific chat
export const getChatMessages = async (chatId: string): Promise<Message[]> => {
  const res = await axios.get(`${API_URL}/messages/${chatId}`, {
    withCredentials: true,
  });
  return res.data;
};

// Create a new chat session
export const createChatSession = async (
  title: string
): Promise<ChatSession> => {
  const res = await axios.post(
    `${API_URL}/sessions`,
    { title },
    { withCredentials: true }
  );
  return res.data;
};

// Send a new message to a chat
export const sendMessage = async (
  chatId: string,
  text: string
): Promise<Message> => {
  const res = await axios.post(
    `${API_URL}/send`,
    { chatId, text },
    { withCredentials: true }
  );
  return res.data;
};
