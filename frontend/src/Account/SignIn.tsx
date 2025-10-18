import { useEffect, useState } from "react";
import { FiMail, FiLock, FiEye, FiEyeOff } from "react-icons/fi";
import { useNavigate } from "react-router-dom";
import { signIn } from "../services/AuthService";
import { getUserConnections } from "../services/DatabaseService";

export default function SignIn() {
  const [showPassword, setShowPassword] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    if (error) {
      // Set a timer to hide the error after 3 seconds (3000ms)
      const timer = setTimeout(() => {
        setError(""); // Clear the error after 3 seconds
      }, 2000);

      // Clean up the timer when the component unmounts or the error changes
      return () => clearTimeout(timer);
    }
  }, [error]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    try {
      const response = await signIn({ email, password });
      if (response.status === 200) {
        const userId = response.data.user.id;

        // Check if user has any database connections
        const connections = await getUserConnections(userId);

        if (connections && connections.length > 0) {
          navigate("/chatinterface", {
            state: { successMessage: "You are signed in successfully!" },
          });
        } else {
          navigate("/account/databaseconnection", {
            state: { successMessage: "You are signed in successfully!" },
          });
        }
      }
    } catch (error: any) {
      if (error.response) {
        setError(error.response.data || "An error occurred");
      } else if (error.request) {
        setError("Network error, please try again later");
      } else {
        setError("Error: " + error.message);
      }
    }
  };

  return (
    <div className="flex-grow-1 d-flex align-items-center justify-content-center">
      {error && (
        <div
          className={`alert alert-danger fade position-fixed top-0 start-50 translate-middle-x mt-4 ${
            error ? "show" : ""
          }`}
          style={{
            zIndex: 1050,
            boxShadow: "0 2px 10px rgba(0,0,0,0.1)",
            borderRadius: "8px",
            padding: "1rem 2rem",
            minWidth: "300px",
            maxWidth: "90vw",
          }}
          role="alert"
        >
          <div className="d-flex justify-content-center align-items-center">
            <i className="bi bi-check-circle-fill me-2"></i>
            <strong>{error}</strong>
          </div>
        </div>
      )}
      <div className="container">
        <div className="row justify-content-center">
          <div className="col-12 col-md-8 col-lg-5">
            <div className="card border-0 shadow-lg">
              <div className="card-header bg-primary text-white text-center py-4 border-0 rounded-top">
                <h4 className="mb-0 fw-bold">Welcome Back</h4>
                <small className="text-white-50">
                  Please sign in to continue
                </small>
              </div>

              <div className="card-body p-4 p-md-5">
                <form onSubmit={handleSubmit}>
                  <div className="mb-4">
                    <label className="form-label fw-semibold">Email</label>
                    <div className="input-group">
                      <span className="input-group-text bg-light border-end-0">
                        <FiMail className="text-muted" />
                      </span>
                      <input
                        type="email"
                        className="form-control border-start-0 ps-2"
                        placeholder="Enter your email"
                        required
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="mb-4">
                    <label className="form-label fw-semibold">Password</label>
                    <div className="input-group">
                      <span className="input-group-text bg-light border-end-0">
                        <FiLock className="text-muted" />
                      </span>
                      <input
                        type={showPassword ? "text" : "password"}
                        className="form-control border-start-0 border-end-0 ps-2"
                        placeholder="Enter your password"
                        required
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                      />
                      <span
                        className="input-group-text bg-light border-start-0 cursor-pointer"
                        onClick={() => setShowPassword(!showPassword)}
                      >
                        {showPassword ? (
                          <FiEyeOff className="text-muted" />
                        ) : (
                          <FiEye className="text-muted" />
                        )}
                      </span>
                    </div>
                  </div>

                  <button
                    type="submit"
                    className="btn btn-primary w-100 py-2 mb-4 fw-semibold"
                  >
                    Sign In
                  </button>

                  <p className="text-center">
                    Don't have an account?{" "}
                    <a
                      href="/account/signup"
                      className="text-primary text-decoration-none fw-semibold"
                    >
                      Sign Up
                    </a>
                  </p>
                </form>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
