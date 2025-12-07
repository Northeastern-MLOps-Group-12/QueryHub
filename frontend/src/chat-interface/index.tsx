import React, { useEffect, useState, useRef, type JSX } from "react";
import botLogo from "../../public/logo.png";
import useAuth from "../hooks/useAuth";
import { Spinner } from "react-bootstrap";
import {
  getChatSessions,
  createNewChat,
  getChatHistory,
  sendMessage,
  getAttachmentUrl,
  getVisualizationUrl,
  type ChatSession,
  type Message,
} from "../services/chatService";
import NewChatModal from "./NewChatModal";

export default function ChatInterface(): JSX.Element {
  const { userData, userId } = useAuth();
  console.log("ChatInterface Render - UserData:", userData);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [selectedChat, setSelectedChat] = useState<ChatSession | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [newMessage, setNewMessage] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // ICONS
  const SendIcon = () => <i className="bi bi-send-fill"></i>;
  const MenuIcon = () => <i className="bi bi-list"></i>;

  // New Chat Modal state
  const [showNewChatModal, setShowNewChatModal] = useState(false);

  // Refs
  const messagesContainerRef = useRef<HTMLDivElement | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const leftListRef = useRef<HTMLDivElement | null>(null);

  // Caching helper functions
  const cacheKeySessions = `chat_sessions_${userData?.user_id}`;
  const cacheKeyMessages = (chatId: string) => `chat_messages_${chatId}`;

  // Load cached data from localStorage
  const loadCachedSessions = (): ChatSession[] | null => {
    const cached = localStorage.getItem(cacheKeySessions);
    return cached ? JSON.parse(cached) : null;
  };

  // Save data to localStorage
  const saveCachedSessions = (sessions: ChatSession[]) => {
    localStorage.setItem(cacheKeySessions, JSON.stringify(sessions));
  };

  // Load cached messages for a specific chat
  const loadCachedMessages = (chatId: string): Message[] | null => {
    const cached = localStorage.getItem(cacheKeyMessages(chatId));
    return cached ? JSON.parse(cached) : null;
  };

  // Save messages for a specific chat
  const saveCachedMessages = (chatId: string, messages: Message[]) => {
    localStorage.setItem(cacheKeyMessages(chatId), JSON.stringify(messages));
  };

  // Load chat sessions on mount
  useEffect(() => {
    if (!userData?.user_id) return;
    const cachedSessions = loadCachedSessions();
    if (cachedSessions) {
      setSessions(cachedSessions);
      if (cachedSessions.length > 0) {
        setSelectedChat(cachedSessions[0]);
        const cachedMsgs = loadCachedMessages(cachedSessions[0].chat_id);
        if (cachedMsgs) setMessages(cachedMsgs);
      }
    }

    // Fetch from backend
    (async () => {
      try {
        const chats = await getChatSessions();
        setSessions(chats);
        saveCachedSessions(chats);
        if (chats.length > 0) {
          setSelectedChat(chats[0]);
          const msgs = await getChatHistory(chats[0].chat_id);
          setMessages(msgs.history);
          saveCachedMessages(chats[0].chat_id, msgs.history);
        }
      } catch (err) {
        console.error("Error loading chats:", err);
      }
    })();
  }, [userData]);

  // Load messages when a chat is selected
  const handleSelectChat = async (chat: ChatSession) => {
    try {
      setSelectedChat(chat);
      const cachedMsgs = loadCachedMessages(chat.chat_id);
      if (cachedMsgs) {
        setMessages(cachedMsgs);
      } else {
        const msgs = await getChatHistory(chat.chat_id);
        setMessages(msgs.history);
        saveCachedMessages(chat.chat_id, msgs.history);
      }
    } catch (err) {
      console.error("Error loading messages:", err);
    }
  };

  // Open modal
  const handleNewChat = () => {
    setShowNewChatModal(true);
  };

  // Confirm create chat from modal (backend returns no title)
  const handleConfirmCreateChat = async (titleFromUser: string) => {
    setShowNewChatModal(false);
    if (!userData?.user_id) return;
    try {
      const newChat = await createNewChat(titleFromUser.trim() || "New Chat");
      const fallbackTitle =
        titleFromUser?.trim() || `Untitled Chat ${sessions.length + 1}`;
      const chatWithTitle: ChatSession = {
        ...newChat,
        chat_title: fallbackTitle,
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

  // Cancel create chat
  const handleCancelCreateChat = () => {
    setShowNewChatModal(false);
  };

  const handleSendMessage = async () => {
    // 1. Validation
    if (!newMessage.trim() || !selectedChat) return;

    const textToSend = newMessage.trim();
    setNewMessage("");

    // 2. Create the User's Message object optimistically
    const userMsg: Message = {
      message_id: `temp-${Date.now()}`, // Temporary ID until refresh
      sender: "user",
      created_at: new Date().toISOString(),
      content: { text: textToSend }
    };

    // 3. Update State & Cache with User Message immediately
    const messagesWithUser = [...messages, userMsg];
    setMessages(messagesWithUser);
    saveCachedMessages(selectedChat.chat_id, messagesWithUser); // Update cache if you have this function

    try {
      // 4. Call API (returns only the Bot's response)
      const botMsg = await sendMessage(selectedChat.chat_id, textToSend);

      // 5. Update State & Cache with Bot Message
      setMessages((prevMessages) => {
        const updatedList = [...prevMessages, botMsg];
        // Save to cache inside here to ensure we have the full list
        saveCachedMessages(selectedChat.chat_id, updatedList); 
        return updatedList;
      });

      // 6. UPDATE & RE-SORT SESSIONS LIST [FIX IS HERE]
      setSessions((prevSessions) => {
        // A. Update the timestamp of the current chat
        const updatedList = prevSessions.map((s) =>
          s.chat_id === selectedChat.chat_id 
            ? { ...s, updated_at: new Date().toISOString() } 
            : s
        );

        // B. Sort the list immediately (Newest first)
        const sortedList = updatedList.sort((a, b) => 
          new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
        );
        
        // C. Update Cache
        saveCachedSessions(sortedList);
        
        return sortedList;
      });

    } catch (err) {
      console.error("Error sending message:", err);
      // Optional: Add logic to remove the user message or show an error state
      alert("Failed to send message");
    }
  };

  // Handle Enter key press in input
  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Scroll to bottom on messages update
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    // Only scroll if the content is taller than the container
    if (container.scrollHeight > container.clientHeight) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  // Mobile sidebar handling
  useEffect(() => {
    if (window.innerWidth <= 768) setSidebarOpen(false);
    const handleResize = () => {
      if (window.innerWidth <= 768) setSidebarOpen(false);
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const FileAttachment = ({ msgId, fileName }: { msgId: string; fileName: string }) => {
    const [loading, setLoading] = useState(false);

    const handleDownload = async () => {
      try {
        setLoading(true);
        // 1. Fetch the signed URL from your backend
        const signedUrl = await getAttachmentUrl(msgId);
        
        // 2. Trigger download (create invisible link)
        const link = document.createElement("a");
        link.href = signedUrl;
        link.download = fileName || "download.csv"; // Optional: suggest filename
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      } catch (err) {
        console.error("Failed to download:", err);
        alert("Failed to download file.");
      } finally {
        setLoading(false);
      }
    };

    return (
      <div 
        className="d-flex align-items-center gap-2 p-2 mt-2 bg-white rounded border cursor-pointer hover-shadow"
        style={{ cursor: "pointer", maxWidth: "100%" }}
        onClick={handleDownload}
      >
        <div className="bg-light p-2 rounded text-primary">
          {loading ? <Spinner size="sm" animation="border" /> : <i className="bi bi-file-earmark-text-fill fs-5"></i>}
        </div>
        <div className="overflow-hidden">
          <div className="fw-semibold text-truncate small">Generated Data</div>
          <div className="text-muted small text-truncate" style={{fontSize: "0.75rem"}}>
              {fileName || "data.csv"}
          </div>
        </div>
        <i className="bi bi-download ms-auto text-secondary"></i>
      </div>
    );
  };

  const VisualizationButton = ({ msgId }: { msgId: string }) => {
    const [loading, setLoading] = useState(false);

    const handleViewDashboard = async () => {
      try {
        setLoading(true);
        const signedUrl = await getVisualizationUrl(msgId);
        // Open in new tab
        window.open(signedUrl, "_blank");
      } catch (err) {
        console.error("Failed to open dashboard:", err);
        alert("Failed to load visualization.");
      } finally {
        setLoading(false);
      }
    };

    return (
      <button
        onClick={handleViewDashboard}
        disabled={loading}
        className="btn btn-sm btn-outline-primary w-100 mt-2 d-flex align-items-center justify-content-center gap-2"
      >
        {loading ? <Spinner size="sm" animation="border" /> : <i className="bi bi-bar-chart-fill"></i>}
        View Dashboard
      </button>
    );
  };

  return (
    <div className="d-flex mt-2" style={{ height: "calc(100vh - 60px)", overflow: "hidden" }}>
      {/* NEW CHAT MODAL */}
      <NewChatModal
        show={showNewChatModal}
        onCancel={handleCancelCreateChat}
        onConfirm={handleConfirmCreateChat}
      />

      {/* LEFT SIDEBAR */}
      <aside
        className="d-none d-md-flex flex-column bg-body-tertiary border-end"
        style={{
          flex: sidebarOpen ? "0 0 300px" : "0 0 0",
          maxWidth: sidebarOpen ? 400 : 0,
          minWidth: sidebarOpen ? 260 : 0,
          overflow: "hidden",
          transition: "all 0.2s ease"
        }}
      >
        <div className="pt-2 px-2 ms-2 border-bottom d-flex justify-content-between align-items-center bg-body-tertiary" style={{paddingBottom: "0.7rem"}}>
          <button
            className="btn btn-primary d-flex align-items-center gap-2 px-3 py-2 rounded-pill shadow-sm"
            onClick={handleNewChat}
            aria-label="New chat"
            style={{ fontWeight: 500, fontSize: "0.9rem"}}
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
        
        {/* Chat sessions list */}
        <div ref={leftListRef} className="flex-grow-1 overflow-auto">
          {sessions.length === 0 ? (
            <p className="text-muted small p-3">No Previous Chats</p>
          ) : (
            <div className="list-group list-group-flush border-0">
              {sessions.map((chat) => (
                <button
                  key={chat.chat_id}
                  onClick={() => handleSelectChat(chat)}
                  className={`list-group-item list-group-item-action border-0 text-start ${
                    selectedChat?.chat_id === chat.chat_id ? "active" : ""
                  }`}
                  type="button"
                >
                  <div className="d-flex align-items-center">
                    <div className="overflow-hidden">
                      <div className="fw-semibold text-truncate">
                        {chat.chat_title}
                      </div>
                      {/* <div className="small text-truncate opacity-75">
                        {chat.lastMessage}
                      </div> */}
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

        {/* Chat sessions list */}
        <div ref={leftListRef} className="flex-grow-1 overflow-auto p-2">
          {sessions.length === 0 ? (
            <p className="text-muted small p-3">No Previous Chats</p>
          ) : (
            <div className="list-group list-group-flush border-0">
              {sessions.map((chat) => (
                <button
                  key={chat.chat_id}
                  onClick={() => {
                    handleSelectChat(chat);
                    setSidebarOpen(false);
                  }}
                  className={`list-group-item list-group-item-action border-0 text-start ${
                    selectedChat?.chat_id === chat.chat_id ? "active" : ""
                  }`}
                  type="button"
                >
                  <div className="d-flex align-items-center">
                    <div className="overflow-hidden">
                      <div className="fw-semibold text-truncate">
                        {chat.chat_title}
                      </div>
                      {/* <div className="small text-truncate opacity-75">
                        {chat.lastMessage}
                      </div> */}
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

      {/* Chat panel */}
      <main
        className="d-flex flex-column flex-grow-1"
        style={{ minWidth: 0, minHeight: 0 }}
      >
        <div
          className="p-3 pb-md-4 border-bottom d-flex align-items-center bg-body-tertiary"
          // style={{ position: "sticky", top: "66px", zIndex: 5 }}
        >

          {/* Mobile menu button */}
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
            {selectedChat?.chat_title || "New Chat"}
          </h2>
        </div>

        {/* Messages container */}
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
                    key={msg.message_id}
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
                            (userData?.first_name
                              ? `https://ui-avatars.com/api/?name=${encodeURIComponent(
                                  userData.first_name
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
                    {/* Message Bubble */}
                    <div
                      className={`p-3 rounded-4 shadow-sm ${
                        msg.sender === "user"
                          ? "bg-primary text-white"
                          : "bg-light text-dark border"
                      }`}
                      style={{
                        maxWidth: "75%",
                        borderBottomRightRadius: msg.sender === "user" ? 0 : undefined,
                        borderBottomLeftRadius: msg.sender === "bot" ? 0 : undefined,
                      }}
                    >
                      {/* 1. TEXT CONTENT */}
                      <div style={{ whiteSpace: "pre-wrap", fontWeight: `${msg.sender === "bot" ? "bold" : "normal"}` }}>
                        {msg.sender === "user" ? msg.content.text : "I found the following results:"}
                      </div>

                      {/* 2. SQL QUERY (Optional: Show if you want debugging) */}
                      {msg.sender === "bot" && msg.content.query && (
                        <div className="mt-2 p-2 bg-dark bg-opacity-10 rounded font-monospace small text-muted">
                          <code>{msg.content.query}</code>
                        </div>
                      )}

                      {/* 3. ATTACHMENT SECTION */}
                      {msg.sender === "bot" && msg.content.attachment?.has_attachment && (
                        <FileAttachment 
                          msgId={msg.message_id} 
                          fileName={msg.content.attachment.file_name || "data.csv"} // Pass filename if available in your JSON
                        />
                      )}

                      {/* 4. VISUALIZATION SECTION */}
                      {msg.sender === "bot" && msg.content.visualization?.has_visualization && (
                        <VisualizationButton msgId={msg.message_id} />
                      )}
                    </div>
                  </div>
                ))}
              </div>
              <div ref={messagesEndRef} />
            </div>

            {/* Message input area */}
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
