import React, { useEffect, useState, useRef, type JSX } from "react";
import botLogo from "../../public/logo.png";
import useAuth from "../Account/UseAuth";

// --- ICONS ---
const SendIcon = () => <i className="bi bi-send-fill"></i>;
const MenuIcon = () => <i className="bi bi-list"></i>;

interface ChatSession {
  id: string;
  title: string;
  lastMessage?: string;
  avatar?: React.ReactNode;
}

interface Message {
  id: string;
  sender: "user" | "bot";
  text: string;
  timestamp: string;
}

export default function DemoInterface(): JSX.Element {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [selectedChat, setSelectedChat] = useState<ChatSession | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [newMessage, setNewMessage] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const { userData } = useAuth();

  const messagesContainerRef = useRef<HTMLDivElement | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const leftListRef = useRef<HTMLDivElement | null>(null);

  // --- MOCK DATA ---
  const mockSessions: ChatSession[] = [
    {
      id: "1",
      title: "SQL Query Assistant",
      lastMessage: "Sure, let's refine that JOIN statement...",
    },
    {
      id: "2",
      title: "Data Cleaning Help",
      lastMessage: "You can use Pandas for that task.",
    },
    {
      id: "3",
      title: "RAG Chatbot Project",
      lastMessage: "Let’s connect your dataset first.",
    },
    {
      id: "4",
      title: "Marketing Analysis",
      lastMessage: "The latest campaign data looks promising.",
    },
    {
      id: "5",
      title: "Sales Data Insights",
      lastMessage: "Q3 sales have increased by 15%.",
    },
    {
      id: "6",
      title: "Customer Feedback Review",
      lastMessage: "Most customers are satisfied with the new features.",
    },
    {
      id: "7",
      title: "Product Development Ideas",
      lastMessage: "We should consider adding more AI features.",
    },
    {
      id: "8",
      title: "Financial Report Analysis",
      lastMessage: "The revenue growth is steady this quarter.",
    },
    {
      id: "9",
      title: "Website Traffic Discussion",
      lastMessage: "Traffic spiked after the recent blog post.",
    },
    {
      id: "10",
      title: "Social Media Strategy",
      lastMessage: "Engagement rates are up by 20% this month.",
    },
    {
      id: "11",
      title: "Customer Retention Strategies",
      lastMessage: "We need to focus on our loyal customer base.",
    },
    {
      id: "12",
      title: "Inventory Management",
      lastMessage: "Stock levels are optimal for the holiday season.",
    },
    {
      id: "13",
      title: "Supply Chain Optimization",
      lastMessage: "We should streamline our logistics for better efficiency.",
    },
    {
      id: "14",
      title: "HR Policies Update",
      lastMessage: "The new remote work policy has been well received.",
    },
    {
      id: "15",
      title: "IT Infrastructure Planning",
      lastMessage: "We need to upgrade our servers for better performance.",
    },
  ];

  const mockMessages: Record<string, Message[]> = {
    "1": [
      {
        id: "m1",
        sender: "user",
        text: "How do I write a SQL query to join two tables?",
        timestamp: "2025-10-17T10:00:00Z",
      },
      {
        id: "m2",
        sender: "bot",
        text: "Use the JOIN clause — for example: SELECT * FROM table1 JOIN table2 ON table1.id = table2.ref_id;",
        timestamp: "2025-10-17T10:01:00Z",
      },
      {
        id: "m3",
        sender: "user",
        text: "What about a LEFT JOIN?",
        timestamp: "2025-10-17T10:02:00Z",
      },
      {
        id: "m4",
        sender: "bot",
        text: "A LEFT JOIN returns all records from the left table (table1), and the matched records from the right table (table2). The result is NULL from the right side, if there is no match.",
        timestamp: "2025-10-17T10:03:00Z",
      },
    ],
    "2": [
      {
        id: "m5",
        sender: "user",
        text: "How to remove duplicates in a dataset?",
        timestamp: "2025-10-17T11:00:00Z",
      },
      {
        id: "m6",
        sender: "bot",
        text: "You can use df.drop_duplicates() in Pandas.",
        timestamp: "2025-10-17T11:01:00Z",
      },
      {
        id: "m7",
        sender: "user",
        text: "And how to handle missing values?",
        timestamp: "2025-10-17T11:02:00Z",
      },
      {
        id: "m8",
        sender: "bot",
        text: "You can use df.fillna() to fill missing values or df.dropna() to remove them.",
        timestamp: "2025-10-17T11:03:00Z",
      },
      {
        id: "m9",
        sender: "user",
        text: "What about outliers?",
        timestamp: "2025-10-17T11:04:00Z",
      },
      {
        id: "m10",
        sender: "bot",
        text: "You can use df[~df['column'].isin(outliers)] to remove them.",
        timestamp: "2025-10-17T11:05:00Z",
      },
      {
        id: "m11",
        sender: "user",
        text: "How can I integrate my RAG model here?",
        timestamp: "2025-10-17T12:00:00Z",
      },
      {
        id: "m12",
        sender: "bot",
        text: "First, upload your dataset to the vector store.",
        timestamp: "2025-10-17T12:01:00Z",
      },
      {
        id: "m13",
        sender: "user",
        text: "How do I connect to the vector store?",
        timestamp: "2025-10-17T12:02:00Z",
      },
      {
        id: "m14",
        sender: "bot",
        text: "Use the provided API keys and endpoints to connect.",
        timestamp: "2025-10-17T12:03:00Z",
      },
      {
        id: "m15",
        sender: "user",
        text: "How to query the RAG model?",
        timestamp: "2025-10-17T12:04:00Z",
      },
      {
        id: "m16",
        sender: "bot",
        text: "Send your questions to the model endpoint with the appropriate parameters.",
        timestamp: "2025-10-17T12:05:00Z",
      },
    ],
    "3": [
      {
        id: "m17",
        sender: "user",
        text: "How can I integrate my RAG model here?",
        timestamp: "2025-10-17T12:00:00Z",
      },
      {
        id: "m18",
        sender: "bot",
        text: "First, upload your dataset to the vector store.",
        timestamp: "2025-10-17T12:01:00Z",
      },
    ],
  };

  // --- INIT ---
  useEffect(() => {
    setTimeout(() => {
      setSessions(mockSessions);
      if (mockSessions.length > 0) {
        setSelectedChat(mockSessions[0]);
        setMessages(mockMessages[mockSessions[0].id] || []);
      }
    }, 500);
  }, []);

  // --- MOBILE-FIRST SIDEBAR ---
  useEffect(() => {
    // Close sidebar by default if screen width <= 768px
    if (window.innerWidth <= 768) setSidebarOpen(false);

    const handleResize = () => {
      if (window.innerWidth <= 768) setSidebarOpen(false);
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // --- SCROLL TO BOTTOM ---
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    // Only scroll if the content is taller than the container
    if (container.scrollHeight > container.clientHeight) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  // --- HANDLERS ---
  const handleSelectChat = (chat: ChatSession) => {
    setSelectedChat(chat);
    setMessages(mockMessages[chat.id] || []);
  };

  const handleSendMessage = () => {
    if (!newMessage.trim() || !selectedChat) return;

    const userMsg: Message = {
      id: `user-${Date.now()}`,
      sender: "user",
      text: newMessage.trim(),
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setNewMessage("");

    // Simulate bot response
    setTimeout(() => {
      setMessages((prev) => [
        ...prev,
        {
          id: `bot-${Date.now()}`,
          sender: "bot",
          text: "This is a sample response. I'm processing your request...",
          timestamp: new Date().toISOString(),
        },
      ]);
    }, 1200);
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="d-flex mt-2" style={{ height: "100vh", minHeight: 600 }}>
      {/* LEFT SIDEBAR */}
      {/* Desktop: normal sidebar (pushes layout) */}
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
        <div className="p-2 ps-3 border-bottom d-flex justify-content-between align-items-center bg-body-tertiary">
          <button
            className="btn btn-primary d-flex align-items-center gap-2 px-3 py-2 rounded-pill shadow-sm"
            aria-label="New chat"
            style={{
              fontWeight: 500,
              fontSize: "0.9rem",
              transition: "all 0.2s",
            }}
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
              backgroundColor: "rgba(252, 218, 218, 0.8)",
            }}
          >
            <i
              className="bi bi-x-lg text-danger"
              style={{
                fontSize: "1rem",
                fontWeight: 700,
                WebkitTextStroke: "1px #b30000", // thicker edge
                color: "#cc0000", // darker red
              }}
            ></i>
          </button>
        </div>
        <div ref={leftListRef} className="flex-grow-1 overflow-auto">
          {sessions.length === 0 ? (
            <p className="text-muted small p-3">Loading chats...</p>
          ) : (
            <div className="list-group list-group-flush border-0">
              {sessions.map((chat) => (
                <button
                  key={chat.id}
                  data-chat-id={chat.id}
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

      {/* Mobile off-canvas: overlays right panel when open */}
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
        {/* Header */}
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
            aria-label="New chat"
            style={{
              fontWeight: 500,
              fontSize: "0.9rem",
              transition: "all 0.2s",
            }}
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
              backgroundColor: "rgba(252, 218, 218, 0.8)",
            }}
          >
            <i
              className="bi bi-x-lg text-danger"
              style={{
                fontSize: "1rem",
                fontWeight: 700,
                WebkitTextStroke: "1px #b30000", // thicker edge
                color: "#cc0000", // darker red
              }}
            ></i>
          </button>
        </div>

        {/* Scrollable chat list */}
        <div ref={leftListRef} className="flex-grow-1 overflow-auto p-2">
          {sessions.length === 0 ? (
            <p className="text-muted small p-3">Loading chats...</p>
          ) : (
            <div className="list-group list-group-flush border-0">
              {sessions.map((chat) => (
                <button
                  key={chat.id}
                  data-chat-id={chat.id}
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

      {/* Mobile backdrop when off-canvas open */}
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
        {/* HEADER */}
        <div
          className="p-3 pb-md-4 border-bottom d-flex align-items-center bg-body-tertiary"
          style={{
            position: "sticky",
            top: "66px",
            zIndex: 5,
          }}
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
          <h2 className="h6 m-0 fw-semibold">
            {selectedChat?.title || "Select a chat"}
          </h2>
        </div>

        {selectedChat ? (
          <>
            {/* MESSAGES */}
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
                      aria-hidden
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
                            ? "0 2px 6px rgba(0, 0, 0, 0.25)"
                            : "0 2px 5px rgba(0, 0, 0, 0.15)",
                      }}
                    >
                      {msg.text}
                    </div>
                  </div>
                ))}
              </div>
              <div ref={messagesEndRef} />
            </div>

            {/* INPUT */}
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
                  style={{
                    paddingRight: "3rem", // space for button
                    zIndex: 1,
                  }}
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
          <div className="d-flex flex-column align-items-center justify-content-center flex-grow-1 text-muted">
            <i
              className="bi bi-chat-dots-fill"
              style={{ fontSize: "4rem" }}
              aria-hidden
            ></i>
            <p className="mt-3">Select a chat to start messaging</p>
          </div>
        )}
      </main>
    </div>
  );
}
