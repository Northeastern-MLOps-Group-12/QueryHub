import api from "./api"; // Your configured axios instance

// --- Import Static Mock Data ---
// import GetAllChatsData from "./GetAllChats.json";
// import GetChatHistoryData from "./GetChatHistory.json";
// import PostNewChatData from "./PostNewChat.json";
// import PostNewMessageData from "./PostNewMessage.json";
// import GetDataAttachmentData from "./GetDataAttachment.json";
// import GetVizAttachmentData from "./GetVizAttachment.json";

// --- 1. Type Definitions based on API Contract ---

export interface ChatSession {
  chat_id: string;
  chat_title: string;
  created_at: string;
  updated_at: string;
}

export interface AttachmentMetadata {
  has_attachment: boolean;
  file_name?: string;
  file_type?: string;
  file_size_bytes?: number;
}

export interface VisualizationMetadata {
  has_visualization: boolean;
  file_name?: string;
}

export interface MessageContent {
  text: string;
  query?: string; // SQL query (optional, usually for bot messages)
  attachment?: AttachmentMetadata;
  visualization?: VisualizationMetadata;
}

export interface Message {
  message_id: string; // Note: Backend uses 'message_id' in history, 'msg_id' in POST response. handled below.
  sender: "user" | "bot";
  created_at: string;
  content: MessageContent;
}

// Response structure for GET /chats/{chat_id}
export interface ChatHistoryResponse {
  chat_id: string;
  chat_title: string;
  created_at: string;
  updated_at: string;
  history: Message[];
}

// Response structure for GET /chats response wrapper
interface ChatListResponse {
  chats: ChatSession[];
}

export interface SignedUrlResponse {
  url: string;
  expires_in_seconds: number;
}

// --- 2. API Functions (MOCKED) ---

/**
 * GET /chats
 * Lists all chats for the logged-in user.
 */
export async function getChatSessions(): Promise<ChatSession[]> {
  // --- REAL API CALL (Commented Out) ---
  const response = await api.get<ChatListResponse>("/chats/chats");
  return response.data.chats;

  // --- MOCK DATA ---
  // console.log("Using mock data for getChatSessions");
  // return Promise.resolve(GetAllChatsData.chats);
}

/**
 * POST /chats
 * Creates a new chat session.
 */
export async function createNewChat(title: string = "New Chat"): Promise<ChatSession> {
  // --- REAL API CALL (Commented Out) ---
  console.log("Creating new chat with title:", title);
  const response = await api.post<ChatSession>("/chats/chats", {
    chat_title: title,
  });
  return response.data;

  // --- MOCK DATA ---
  // Casting to ChatSession to match interface strictly if JSON is loose
  // return Promise.resolve(PostNewChatData as ChatSession); 
}

/**
 * GET /chats/{chat_id}
 * Loads the full conversation history for a specific chat.
 */
export async function getChatHistory(chatId: string): Promise<ChatHistoryResponse> {
  // --- REAL API CALL (Commented Out) ---
  const response = await api.get<ChatHistoryResponse>(`/chats/chats/${chatId}`);
  return response.data;

  // --- MOCK DATA ---
  // We need to cast the history items because JSON imports might infer slightly different types
  // const mockData = GetChatHistoryData as unknown as ChatHistoryResponse;
  // return Promise.resolve(mockData);
}

/**
 * POST /chats/{chat_id}/messages
 * Sends a user query to the backend and receives the bot's response metadata.
 */
export async function sendMessage(chatId: string, text: string): Promise<Message> {
  // --- REAL API CALL (Commented Out) ---
  const response = await api.post<any>(`/chats/chats/${chatId}/messages`, {
    text: text,
  });
  const data = response.data;
  return {
    ...data,
    message_id: data.message_id || data.msg_id 
  };

  // --- MOCK DATA ---
  // const data = PostNewMessageData;
  
  // // Simulate network delay for better UX testing
  // await new Promise(resolve => setTimeout(resolve, 1000));

  // return Promise.resolve({
  //   ...data,
  //   // Map msg_id to message_id and ensure 'sender' is typed correctly
  //   message_id: data.msg_id,
  //   sender: data.sender as "user" | "bot",
  //   content: data.content
  // } as Message);
}

/**
 * GET /messages/{msg_id}/attachment
 * Gets a signed URL for downloading the CSV/File attachment.
 */
export async function getAttachmentUrl(msgId: string): Promise<string> {
  // --- REAL API CALL (Commented Out) ---
  const response = await api.get<SignedUrlResponse>(`/chats/messages/${msgId}/attachment`);
  return response.data.url;

  // --- MOCK DATA ---
  // return Promise.resolve(GetDataAttachmentData.url);
}

/**
 * GET /messages/{msg_id}/visualization
 * Gets a signed URL for loading the HTML visualization (iframe).
 */
export async function getVisualizationUrl(msgId: string): Promise<string> {
  // --- REAL API CALL (Commented Out) ---
  const response = await api.get<SignedUrlResponse>(`/chats/messages/${msgId}/visualization`);
  return response.data.url;

  // --- MOCK DATA ---
  // return Promise.resolve(GetVizAttachmentData.url);
}