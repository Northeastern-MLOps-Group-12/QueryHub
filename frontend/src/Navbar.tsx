import { Button } from "react-bootstrap";
import useAuth from "./Account/UseAuth";
import { signOut } from "./Services/AuthService";
import { useEffect, useState, useRef } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { FiDatabase, FiLogOut } from "react-icons/fi";

export default function Navbar() {
  const { userId, userData } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [showProfile, setShowProfile] = useState(false);

  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setIsAuthenticated(!!userId);
  }, [userId]);

  // close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setShowProfile(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSignOut = async () => {
    try {
      await signOut();
      setShowProfile(false);
      navigate("/Account/SignIn");
    } catch (error) {
      console.error("Error logging out:", error);
    }
  };

  const handleAuthAction = () => {
    if (!isAuthenticated) navigate("/Account/SignIn");
    else if (location.pathname !== "/ChatInterface") navigate("/ChatInterface");
  };

  const handleScrollTo = (sectionId: string) => {
    if (location.pathname !== "/") navigate(`/#${sectionId}`);
    else {
      const el = document.getElementById(sectionId);
      if (el) el.scrollIntoView({ behavior: "smooth" });
      window.history.replaceState(null, "", `#${sectionId}`);
    }
  };

  const avatarUrl =
    userData?.avatarUrl ||
    (userData?.firstName
      ? `https://ui-avatars.com/api/?name=${encodeURIComponent(
          userData.firstName
        )}&background=random`
      : "/src/assets/default-avatar.png");

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
            <button
              className="nav-link btn btn-link"
              onClick={() => handleScrollTo("about")}
            >
              About
            </button>

            {isAuthenticated ? (
              <div className="position-relative ms-3" ref={menuRef}>
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

                {showProfile && (
                  <div
                    className="position-absolute end-0 mt-2 bg-white shadow rounded-4 p-3"
                    style={{
                      width: "260px",
                      zIndex: 1000,
                      border: "1px solid #eaeaea",
                    }}
                  >
                    <div className="d-flex justify-content-end">
                      <button
                        className="btn btn-sm btn-light p-1 mb-2"
                        onClick={() => setShowProfile(false)}
                        style={{ lineHeight: 0 }}
                      >
                        &#x2715; {/* simple X symbol */}
                      </button>
                    </div>

                    <div className="text-center mb-3">
                      <img
                        src={avatarUrl}
                        alt="User Avatar"
                        className="rounded-circle mb-2"
                        style={{ width: "70px", height: "70px" }}
                      />
                      <h6 className="fw-bold mb-0">
                        {userData?.firstName} {userData?.lastName}
                      </h6>
                      <small className="text-muted">{userData?.email}</small>
                    </div>

                    <hr className="my-2" />

                    <Button
                      variant="outline-primary"
                      className="w-100 mb-2 d-flex align-items-center justify-content-center gap-2"
                      onClick={() => {
                        setShowProfile(false);
                        navigate("/Account/ConnectedDatabases");
                      }}
                    >
                      <FiDatabase /> Connected Databases
                    </Button>

                    <Button
                      variant="outline-danger"
                      className="w-100 d-flex align-items-center justify-content-center gap-2"
                      onClick={handleSignOut}
                    >
                      <FiLogOut /> Sign Out
                    </Button>
                  </div>
                )}
              </div>
            ) : (
              <Button
                variant="primary"
                className="ms-lg-2"
                onClick={handleAuthAction}
              >
                Sign In
              </Button>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
