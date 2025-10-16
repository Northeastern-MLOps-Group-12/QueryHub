import { Button } from "react-bootstrap";
import useAuth from "./Account/UseAuth";
import { signOut } from "./Services/authService";
import { useEffect, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";

export default function Navbar() {
  const { userEmail } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const [isAuthenticated, setIsAuthenticated] = useState(false);

  useEffect(() => {
    setIsAuthenticated(!!userEmail);
  }, [userEmail]);

  const handleSignOut = async () => {
    try {
      await signOut();
      navigate("/Account/SignIn");
    } catch (error) {
      console.error("Error logging out:", error);
    }
  };

  const handleAuthAction = () => {
    if (!isAuthenticated) {
      navigate("/Account/SignIn");
    } else if (location.pathname !== "/ChatInterface") {
      navigate("/ChatInterface");
    }
  };

  const handleScrollTo = (sectionId: string) => {
    if (location.pathname !== "/") {
      // Navigate to homepage with hash
      navigate(`/#${sectionId}`);
    } else {
      // Already on homepage
      const el = document.getElementById(sectionId);
      if (el) el.scrollIntoView({ behavior: "smooth" });
      // Update URL hash without reloading
      window.history.replaceState(null, "", `#${sectionId}`);
    }
  };

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
          aria-controls="navbarNav"
          aria-expanded="false"
          aria-label="Toggle navigation"
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
            {isAuthenticated && location.pathname === "/ChatInterface" ? (
              <>
                <span className="navbar-text me-2">Welcome, {userEmail}</span>
                <Button
                  variant="outline-primary"
                  className="ms-lg-2"
                  onClick={handleSignOut}
                >
                  Sign Out
                </Button>
              </>
            ) : (
              <Button
                variant="primary"
                className="ms-lg-2"
                onClick={handleAuthAction}
              >
                {isAuthenticated ? "Chat" : "Sign In"}
              </Button>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
