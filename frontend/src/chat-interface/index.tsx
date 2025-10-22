import React, { useEffect, useState, useRef, type JSX } from "react";
import botLogo from "../../public/logo.png";
import useAuth from "../hooks/useAuth";
import {
  getChatSessions,
  getChatMessages,
  createNewChat,
  sendMessage,
  type ChatSession,
  type Message,
} from "../services/ChatService";
import NewChatModal from "./NewChatModal";

// --- ICONS ---
const SendIcon = () => <i className="bi bi-send-fill"></i>;
const MenuIcon = () => <i className="bi bi-list"></i>;

export default function ChatInterface(): JSX.Element {
  const { userData } = useAuth();
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [selectedChat, setSelectedChat] = useState<ChatSession | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [newMessage, setNewMessage] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // modal state
  const [showNewChatModal, setShowNewChatModal] = useState(false);

  const messagesContainerRef = useRef<HTMLDivElement | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const leftListRef = useRef<HTMLDivElement | null>(null);

  // --- Helper functions for cache ---
  const cacheKeySessions = `chat_sessions_${userData?.userId}`;
  const cacheKeyMessages = (chatId: string) => `chat_messages_${chatId}`;

  const loadCachedSessions = (): ChatSession[] | null => {
    const cached = localStorage.getItem(cacheKeySessions);
    return cached ? JSON.parse(cached) : null;
  };

  const saveCachedSessions = (sessions: ChatSession[]) => {
    localStorage.setItem(cacheKeySessions, JSON.stringify(sessions));
  };

  const loadCachedMessages = (chatId: string): Message[] | null => {
    const cached = localStorage.getItem(cacheKeyMessages(chatId));
    return cached ? JSON.parse(cached) : null;
  };

  const saveCachedMessages = (chatId: string, messages: Message[]) => {
    localStorage.setItem(cacheKeyMessages(chatId), JSON.stringify(messages));
  };

  // --- Load chat sessions on mount ---
  useEffect(() => {
    if (!userData?.userId) return;
    const cachedSessions = loadCachedSessions();
    if (cachedSessions) {
      setSessions(cachedSessions);
      if (cachedSessions.length > 0) {
        setSelectedChat(cachedSessions[0]);
        const cachedMsgs = loadCachedMessages(cachedSessions[0].id);
        if (cachedMsgs) setMessages(cachedMsgs);
      }
    }

    // --- Fetch from server ---
    (async () => {
      try {
        const chats = await getChatSessions(userData.userId);
        setSessions(chats);
        saveCachedSessions(chats);
        if (chats.length > 0) {
          setSelectedChat(chats[0]);
          const msgs = await getChatMessages(chats[0].id);
          setMessages(msgs);
          saveCachedMessages(chats[0].id, msgs);
        }
      } catch (err) {
        console.error("Error loading chats:", err);
      }
    })();
  }, [userData]);

  // --- Load messages when a chat is selected ---
  const handleSelectChat = async (chat: ChatSession) => {
    try {
      setSelectedChat(chat);
      const cachedMsgs = loadCachedMessages(chat.id);
      if (cachedMsgs) {
        setMessages(cachedMsgs);
      } else {
        const msgs = await getChatMessages(chat.id);
        setMessages(msgs);
        saveCachedMessages(chat.id, msgs);
      }
    } catch (err) {
      console.error("Error loading messages:", err);
    }
  };

  // open modal
  const handleNewChat = () => {
    setShowNewChatModal(true);
  };

  // confirm create chat from modal (backend returns no title)
  const handleConfirmCreateChat = async (titleFromUser: string) => {
    setShowNewChatModal(false);
    if (!userData?.userId) return;
    try {
      const newChat = await createNewChat(userData.userId);
      const fallbackTitle =
        titleFromUser?.trim() || `Untitled Chat ${sessions.length + 1}`;
      const chatWithTitle: ChatSession = {
        ...newChat,
        title: fallbackTitle,
      };
      const updatedSessions = [chatWithTitle, ...sessions];
      setSessions(updatedSessions);
      setSelectedChat(chatWithTitle);
      setMessages([]);
      saveCachedSessions(updatedSessions);
      // close sidebar on small screens so user sees the new chat
      setSidebarOpen(false);
    } catch (err) {
      console.error("Error creating new chat:", err);
    }
  };

  const handleCancelCreateChat = () => {
    setShowNewChatModal(false);
  };

  // --- Send message ---
  const handleSendMessage = async () => {
    if (!newMessage.trim() || !selectedChat || !userData?.userId) return;
    try {
      const updatedMessages = await sendMessage(
        selectedChat.id,
        userData.userId,
        newMessage.trim()
      );
      setMessages(updatedMessages);
      saveCachedMessages(selectedChat.id, updatedMessages);
      setNewMessage("");

      // Update sessions cache as well (for latest message preview)
      const updatedSessions = sessions.map((s) =>
        s.id === selectedChat.id ? { ...s, lastMessage: newMessage.trim() } : s
      );
      setSessions(updatedSessions);
      saveCachedSessions(updatedSessions);
    } catch (err) {
      console.error("Error sending message:", err);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // --- Scroll to bottom on messages update ---
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    // Only scroll if the content is taller than the container
    if (container.scrollHeight > container.clientHeight) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  // --- Mobile-first sidebar handling ---
  useEffect(() => {
    if (window.innerWidth <= 768) setSidebarOpen(false);
    const handleResize = () => {
      if (window.innerWidth <= 768) setSidebarOpen(false);
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <div className="d-flex mt-2" style={{ height: "100vh", minHeight: 600 }}>
      <NewChatModal
        show={showNewChatModal}
        onCancel={handleCancelCreateChat}
        onConfirm={handleConfirmCreateChat}
      />

      {/* LEFT SIDEBAR */}
      <aside
        className="d-none d-md-flex flex-column bg-body-tertiary border-end position-sticky"
        style={{
          flex: sidebarOpen ? "0 0 300px" : "0 0 0",
          maxWidth: sidebarOpen ? 400 : 0,
          minWidth: sidebarOpen ? 260 : 0,
          overflow: "hidden",
          transition: "all 0.2s ease",
          top: "69px",
          height: "calc(100vh - 69px)",
        }}
      >
        <div className="p-2 ms-2 border-bottom d-flex justify-content-between align-items-center bg-body-tertiary">
          <button
            className="btn btn-primary d-flex align-items-center gap-2 px-3 py-2 rounded-pill shadow-sm"
            onClick={handleNewChat}
            aria-label="New chat"
            style={{ fontWeight: 500, fontSize: "0.9rem" }}
          >
            <i className="bi bi-pencil-square"></i>
            <span>New Chat</span>
          </button>
          <button
            className="btn btn-light d-flex align-items-center justify-content-center rounded-circle shadow-sm border-0"
            onClick={() => setSidebarOpen(false)}
            aria-label="Collapse sidebar"
            style={{
              width: 36,
              height: 36,
              backgroundColor: "rgba(252,218,218,0.8)",
            }}
          >
            <i
              className="bi bi-x-lg text-danger"
              style={{
                fontSize: "1rem",
                fontWeight: 700,
                WebkitTextStroke: "1px #b30000",
                color: "#cc0000",
              }}
            ></i>
          </button>
        </div>

        <div ref={leftListRef} className="flex-grow-1 overflow-auto">
          {sessions.length === 0 ? (
            <p className="text-muted small p-3">No Previous Chats</p>
          ) : (
            <div className="list-group list-group-flush border-0">
              {sessions.map((chat) => (
                <button
                  key={chat.id}
                  onClick={() => handleSelectChat(chat)}
                  className={`list-group-item list-group-item-action border-0 text-start ${
                    selectedChat?.id === chat.id ? "active" : ""
                  }`}
                  type="button"
                >
                  <div className="d-flex align-items-center">
                    <div className="overflow-hidden">
                      <div className="fw-semibold text-truncate">
                        {chat.title}
                      </div>
                      <div className="small text-truncate opacity-75">
                        {chat.lastMessage}
                      </div>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </aside>

      {/* Mobile off-canvas */}
      <div
        className="d-md-none position-fixed start-0 h-100 bg-body-tertiary shadow-sm border-end d-flex flex-column"
        style={{
          width: 280,
          transform: sidebarOpen ? "translateX(0)" : "translateX(-100%)",
          transition: "transform 220ms ease",
          zIndex: 1050,
        }}
        aria-hidden={!sidebarOpen}
      >
        <div
          className="p-2 border-bottom d-flex justify-content-between align-items-center"
          style={{
            position: "sticky",
            zIndex: 10,
            background: "var(--bs-body-bg)",
          }}
        >
          <button
            className="btn btn-primary d-flex align-items-center gap-2 px-3 py-2 rounded-pill shadow-sm"
            onClick={handleNewChat}
            aria-label="New chat"
            style={{ fontWeight: 500, fontSize: "0.9rem" }}
          >
            <i className="bi bi-pencil-square"></i>
            <span>New Chat</span>
          </button>
          <button
            className="btn btn-light d-flex align-items-center justify-content-center rounded-circle shadow-sm border-0"
            onClick={() => setSidebarOpen(false)}
            aria-label="Collapse sidebar"
            style={{
              width: 36,
              height: 36,
              backgroundColor: "rgba(252,218,218,0.8)",
            }}
          >
            <i
              className="bi bi-x-lg text-danger"
              style={{
                fontSize: "1rem",
                fontWeight: 700,
                WebkitTextStroke: "1px #b30000",
                color: "#cc0000",
              }}
            ></i>
          </button>
        </div>

        <div ref={leftListRef} className="flex-grow-1 overflow-auto p-2">
          {sessions.length === 0 ? (
            <p className="text-muted small p-3">No Previous Chats</p>
          ) : (
            <div className="list-group list-group-flush border-0">
              {sessions.map((chat) => (
                <button
                  key={chat.id}
                  onClick={() => {
                    handleSelectChat(chat);
                    setSidebarOpen(false);
                  }}
                  className={`list-group-item list-group-item-action border-0 text-start ${
                    selectedChat?.id === chat.id ? "active" : ""
                  }`}
                  type="button"
                >
                  <div className="d-flex align-items-center">
                    <div className="overflow-hidden">
                      <div className="fw-semibold text-truncate">
                        {chat.title}
                      </div>
                      <div className="small text-truncate opacity-75">
                        {chat.lastMessage}
                      </div>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Mobile backdrop */}
      {sidebarOpen && (
        <div
          className="d-md-none position-fixed start-0 w-100 h-100"
          style={{ background: "rgba(0,0,0,0.3)", zIndex: 1040 }}
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* CHAT PANEL */}
      <main
        className="d-flex flex-column flex-grow-1"
        style={{ minWidth: 0, minHeight: 0 }}
      >
        <div
          className="p-3 pb-md-4 border-bottom d-flex align-items-center bg-body-tertiary"
          style={{ position: "sticky", top: "66px", zIndex: 5 }}
        >
          {!sidebarOpen && (
            <button
              className="btn btn-outline-secondary rounded-circle d-flex align-items-center justify-content-center shadow-sm me-2"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open sidebar"
              style={{
                width: "40px",
                height: "40px",
                padding: 0,
                borderWidth: "1.5px",
              }}
            >
              <MenuIcon />
            </button>
          )}
          <h2 className="h6 mb-0 fw-semibold">
            {selectedChat?.title || "New Chat"}
          </h2>
        </div>

        {selectedChat ? (
          <>
            <div
              ref={messagesContainerRef}
              className="flex-grow-1 overflow-auto d-flex flex-column"
              style={{
                backgroundColor: "var(--bs-secondary-bg)",
                padding: "1rem",
              }}
            >
              <div className="d-flex flex-column gap-4 flex-grow-1">
                {messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`d-flex align-items-end gap-2 ${
                      msg.sender === "user" ? "flex-row-reverse" : ""
                    }`}
                  >
                    <div
                      className={`rounded-circle d-flex align-items-center justify-content-center text-white ${
                        msg.sender === "bot" ? "bg-primary" : "bg-secondary"
                      }`}
                      style={{ width: 40, height: 40 }}
                    >
                      {msg.sender === "bot" ? (
                        <img
                          src={botLogo}
                          alt="Bot Avatar"
                          style={{
                            width: 36,
                            height: 36,
                            borderRadius: "50%",
                            objectFit: "cover",
                            backgroundColor: "#f8f9fa",
                            border: "1px solid #dee2e6",
                          }}
                        />
                      ) : (
                        <img
                          src={
                            userData?.avatarUrl ||
                            (userData?.firstName
                              ? `https://ui-avatars.com/api/?name=${encodeURIComponent(
                                  userData.firstName
                                )}&background=random`
                              : "/src/assets/default-avatar.png")
                          }
                          alt="User Avatar"
                          style={{
                            width: 36,
                            height: 36,
                            borderRadius: "50%",
                            objectFit: "cover",
                            border: "1px solid #dee2e6",
                          }}
                        />
                      )}
                    </div>
                    <div
                      className={`p-3 rounded-4 shadow-sm ${
                        msg.sender === "user"
                          ? "bg-primary text-white"
                          : "bg-light text-dark border"
                      }`}
                      style={{
                        maxWidth: "75%",
                        borderBottomRightRadius:
                          msg.sender === "user" ? 0 : undefined,
                        borderBottomLeftRadius:
                          msg.sender === "bot" ? 0 : undefined,
                        padding: "0.85rem 1.1rem",
                        lineHeight: "1.5",
                        wordBreak: "break-word",
                        boxShadow:
                          msg.sender === "user"
                            ? "0 2px 6px rgba(0,0,0,0.25)"
                            : "0 2px 5px rgba(0,0,0,0.15)",
                      }}
                    >
                      {msg.text}
                    </div>
                  </div>
                ))}
              </div>
              <div ref={messagesEndRef} />
            </div>

            <div
              className="p-3 bg-body-tertiary border-top position-sticky"
              style={{ bottom: 0, zIndex: 20 }}
            >
              <div
                className="position-relative"
                style={{ display: "flex", alignItems: "center" }}
              >
                <input
                  type="text"
                  value={newMessage}
                  onChange={(e) => setNewMessage(e.target.value)}
                  onKeyDown={handleKeyPress}
                  placeholder="Type your message here..."
                  className="form-control form-control-lg rounded-pill"
                  style={{ paddingRight: "3rem", zIndex: 1 }}
                />
                <button
                  type="button"
                  onClick={handleSendMessage}
                  className="btn btn-primary rounded-circle d-flex align-items-center justify-content-center shadow-sm"
                  style={{
                    width: 32,
                    height: 32,
                    position: "absolute",
                    right: "0.75rem",
                    top: "50%",
                    transform: "translateY(-50%)",
                    zIndex: 2,
                  }}
                  disabled={!newMessage.trim()}
                  aria-label="Send"
                >
                  <SendIcon />
                </button>
              </div>
            </div>
          </>
        ) : (
          <div className="d-flex flex-column align-items-center justify-content-center flex-grow-1">
            <p className="text-muted">
              Select or create a chat to start messaging.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
