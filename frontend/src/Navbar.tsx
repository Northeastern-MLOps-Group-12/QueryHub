import useAuth from "./Account/UseAuth";
import axios from "axios";
import { useEffect, useState } from "react";

export default function Navbar() {
  const { userEmail } = useAuth();
  console.log("email in Navbar", userEmail);

  const [isAuthenticated, setIsAuthenticated] = useState(false);

  useEffect(() => {
    console.log("Auth state in Navbar:", { isAuthenticated, userEmail });
  }, [isAuthenticated, userEmail]);

  useEffect(() => {
    if (userEmail) {
      setIsAuthenticated(true);
    } else {
      setIsAuthenticated(false);
    }
  }, [userEmail]);

  const handleSignOut = async () => {
    try {
      const response = await axios.post(
        "http://localhost:8080/api/user/logout",
        null,
        {
          withCredentials: true,
        }
      );
      console.log(response.data);
      window.location.href = "/Account/SignIn";
    } catch (error) {
      console.error("Error logging out", error);
    }
  };

  return (
    <nav className="p-3 navbar navbar-expand-lg navbar-light bg-dark shadow-sm">
      <div className="container-fluid d-flex align-items-center">
        <a
          className="navbar-brand text-white fw-bold 
            fs-3 fs-md-5 fs-lg-4 
            me-2 me-md-3"
          href="/"
          style={{
            whiteSpace: "nowrap",
            lineHeight: "1.2",
          }}
        >
          QueryHub
        </a>

        <div className="ms-auto d-flex align-items-center">
          <a
            href="/"
            className="pt-1 pb-1 pe-2 ps-2 btn btn-outline-light me-2 d-none d-lg-block"
          >
            Home
          </a>

          <a
            href="/ChatInterface"
            className="pt-1 pb-1 pe-2 ps-2 btn btn-light"
          >
            Sign In
          </a>
          {userEmail && (
            // If authenticated, show the "Sign Out" button
            <button
              onClick={handleSignOut}
              className="ms-2 pt-1 pb-1 pe-2 ps-2 btn btn-outline-light"
            >
              Sign Out
            </button>
          )}
        </div>
      </div>
    </nav>
  );
}
