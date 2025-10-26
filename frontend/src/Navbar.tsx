import { Button, Modal } from "react-bootstrap";
import useAuth from "./hooks/useAuth";
import { signOut } from "./services/AuthService";
import { useEffect, useState, useRef } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { FiLogOut } from "react-icons/fi";

// Navigation Bar Component
export default function Navbar() {
  const { userId, userData } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  // State management
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [showProfile, setShowProfile] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  const menuRef = useRef<HTMLDivElement>(null);

  // Detect mobile screen
  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 768);
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Update authentication status
  useEffect(() => {
    setIsAuthenticated(!!userId);
  }, [userId]);

  // Close dropdown when clicking outside (desktop only)
  useEffect(() => {
    if (isMobile) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setShowProfile(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isMobile]);

  // Handle user sign out
  const handleSignOut = async () => {
    try {
      await signOut();
      setShowProfile(false);
      navigate("/account/signin");
    } catch (error) {
      console.error("Error logging out:", error);
    }
  };

  // Scroll to section or navigate to home with hash
  const handleScrollTo = (sectionId: string) => {
    if (location.pathname !== "/") navigate(`/#${sectionId}`);
    else {
      const el = document.getElementById(sectionId);
      if (el) el.scrollIntoView({ behavior: "smooth" });
      window.history.replaceState(null, "", `#${sectionId}`);
    }
  };

  // Determine avatar URL
  const avatarUrl =
    userData?.avatarUrl ||
    (userData?.firstName
      ? `https://ui-avatars.com/api/?name=${encodeURIComponent(
          userData.firstName
        )}&background=random`
      : "/src/assets/default-avatar.png");

  // Profile content inside dropdown or modal
  const ProfileContent = () => (
    <div className="text-center">
      {/* User Avatar and Info */}
      <img
        src={avatarUrl}
        alt="User Avatar"
        className="rounded-circle mb-3 shadow-sm"
        style={{ width: "90px", height: "90px", objectFit: "cover" }}
      />
      <h5 className="fw-bold mb-1">
        {userData?.firstName} {userData?.lastName}
      </h5>
      <p className="text-muted mb-3">{userData?.email}</p>

      {/* Sign Out Button */}
      <Button
        variant="outline-danger"
        className="w-100"
        onClick={handleSignOut}
      >
        <FiLogOut className="me-2" /> Sign Out
      </Button>
    </div>
  );

  return (
    <nav className="p-2 navbar navbar-expand-lg navbar-light bg-white shadow-sm fixed-top">
      <div className="container">
        <a className="navbar-brand text-primary fw-bold" href="/">
          <img
            src={"/logo.png"}
            alt="QueryHub Logo"
            style={{ height: "40px", marginRight: "8px" }}
          />
          QueryHub
        </a>

        {/* Toggler for mobile view */}
        <button
          className="navbar-toggler"
          type="button"
          data-bs-toggle="collapse"
          data-bs-target="#navbarNav"
        >
          <span className="navbar-toggler-icon"></span>
        </button>

        <div className="collapse navbar-collapse" id="navbarNav">
          <div className="navbar-nav ms-auto align-items-center">
            {!isAuthenticated ? (
              <>
              {/* Navigation buttons for unauthenticated users */}
                <button
                  className="nav-link btn btn-link"
                  onClick={() => handleScrollTo("how-it-works")}
                >
                  How It Works
                </button>
                <button
                  className="nav-link btn btn-link"
                  onClick={() => handleScrollTo("features")}
                >
                  Features
                </button>
                <Button
                  variant="primary"
                  className="ms-lg-2"
                  onClick={() => navigate("/account/signin")}
                >
                  Sign In
                </Button>
              </>
            ) : (
              <>
              {/* Navigation buttons for authenticated users */}
                <button
                  className="nav-link btn btn-link"
                  onClick={() => navigate("/chatinterface")}
                >
                  Chat Interface
                </button>
                <button
                  className="nav-link btn btn-link"
                  onClick={() => navigate("/database/connecteddatabases")}
                >
                  Database Connections
                </button>

                {/* Profile Dropdown / Modal */}
                {isMobile ? (
                  <>
                    <Button
                      variant="link"
                      className="p-0 ms-3"
                      onClick={() => setShowProfile(true)}
                    >
                      <img
                        src={avatarUrl}
                        alt="User Avatar"
                        className="rounded-circle"
                        style={{
                          width: "42px",
                          height: "42px",
                          border: "2px solid #007bff",
                        }}
                      />
                    </Button>

                    <Modal
                      show={showProfile}
                      onHide={() => setShowProfile(false)}
                      centered
                      contentClassName="rounded-4 shadow-lg"
                    >
                      <div style={{ position: "relative" }}>
                        {/* Custom close button */}
                        <button
                          className="btn btn-light d-flex align-items-center justify-content-center rounded-circle shadow-sm border-0"
                          onClick={() => setShowProfile(false)}
                          aria-label="Close"
                          style={{
                            width: 36,
                            height: 36,
                            backgroundColor: "rgba(252,218,218,0.8)",
                            position: "absolute",
                            top: 12,
                            right: 12,
                            zIndex: 10,
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
                        
                        {/* Modal Body with Profile Content */}
                        <Modal.Body className="pt-5">
                          <ProfileContent />
                        </Modal.Body>
                      </div>
                    </Modal>
                  </>
                ) : (
                  <div className="position-relative ms-3" ref={menuRef}>
                    {/* Desktop Profile Dropdown */}
                    <img
                      src={avatarUrl}
                      alt="User Avatar"
                      className="rounded-circle"
                      style={{
                        width: "42px",
                        height: "42px",
                        cursor: "pointer",
                        border: "2px solid #007bff",
                      }}
                      onClick={() => setShowProfile((prev) => !prev)}
                    />

                    {/* Profile Dropdown Content */}
                    {showProfile && (
                      <div
                        className="position-absolute end-0 mt-2 bg-white shadow-lg rounded-4 p-3"
                        style={{
                          width: "280px",
                          zIndex: 1000,
                          border: "1px solid #eaeaea",
                        }}
                      >
                        <div className="d-flex justify-content-end">
                          <button
                            className="btn btn-light d-flex align-items-center justify-content-center rounded-circle shadow-sm border-0"
                            onClick={() => setShowProfile(false)}
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
                        <ProfileContent />
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
