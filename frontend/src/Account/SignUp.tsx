import { useEffect, useState } from "react";
import { FiUser, FiMail, FiLock, FiEye, FiEyeOff } from "react-icons/fi";
import { useNavigate } from "react-router-dom";
import { register } from "../Services/AuthService";

export default function SignUp() {
  const navigate = useNavigate();
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [formData, setFormData] = useState({
    firstName: "",
    lastName: "",
    email: "",
    password: "",
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => {
        setError("");
      }, 3000);

      return () => clearTimeout(timer);
    }
  }, [error]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    try {
      const response = await register(formData);
      if (response.status === 201) {
        navigate("/ChatInterface", {
          state: { successMessage: "You are signed in successfully!" },
        });
      }
    } catch (error: any) {
      if (error.response) {
        if (
          error.response.status === 400 &&
          error.response.data === "Email already exists"
        ) {
          setError(
            "This email is already registered. Please use a different email."
          );
        } else {
          setError(error.response.data || "An error occurred");
        }
      } else if (error.request) {
        setError("Network error, please try again later");
      } else {
        setError("Error: " + error.message);
      }
    }
  };

  const [isFocused, setIsFocused] = useState(false);
  const handleFocus = () => {
    setIsFocused(true);
  };

  const handleBlur = () => {
    setIsFocused(false);
  };

  return (
    <div className="flex-grow-1 d-flex align-items-center justify-content-center py-4">
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
      <div className="container-fluid">
        <div className="row justify-content-center">
          <div className="col-12 col-md-6 col-lg-4">
            <div className="card border-0 shadow-lg">
              <div className="card-header bg-primary text-white text-center py-4 border-0 rounded-top">
                <h4 className="mb-0 fw-bold">Create an Account</h4>
                <small className="text-white-50">
                  Please fill in the details to sign up
                </small>
              </div>

              <div className="card-body p-4 p-md-5">
                <form onSubmit={handleSubmit}>
                  <div className="mb-4">
                    <label className="form-label fw-semibold">First Name</label>
                    <div className="input-group">
                      <span className="input-group-text bg-light border-end-0">
                        <FiUser className="text-muted" />
                      </span>
                      <input
                        type="text"
                        name="firstName"
                        value={formData.firstName}
                        onChange={handleChange}
                        className="form-control border-start-0 ps-2"
                        placeholder="Enter your first name"
                        required
                      />
                    </div>
                  </div>

                  <div className="mb-4">
                    <label className="form-label fw-semibold">Last Name</label>
                    <div className="input-group">
                      <span className="input-group-text bg-light border-end-0">
                        <FiUser className="text-muted" />
                      </span>
                      <input
                        type="text"
                        name="lastName"
                        value={formData.lastName}
                        onChange={handleChange}
                        className="form-control border-start-0 ps-2"
                        placeholder="Enter your last name"
                        required
                      />
                    </div>
                  </div>

                  <div className="mb-4">
                    <label className="form-label fw-semibold">Email</label>
                    <div className="input-group">
                      <span className="input-group-text bg-light border-end-0">
                        <FiMail className="text-muted" />
                      </span>
                      <input
                        type="email"
                        name="email"
                        value={formData.email}
                        onChange={handleChange}
                        onFocus={handleFocus}
                        onBlur={handleBlur}
                        className="form-control border-start-0 ps-2"
                        placeholder="Enter your email"
                        required
                        pattern="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
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
                        name="password"
                        value={formData.password}
                        onChange={handleChange}
                        onFocus={handleFocus}
                        onBlur={handleBlur}
                        className="form-control border-start-0 border-end-0 ps-2"
                        placeholder="Enter your password"
                        required
                        pattern="(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*]).{8,}"
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
                    {isFocused && (
                      <div className="mt-2 text-muted">
                        <p>Password must meet the following criteria:</p>
                        <ul>
                          <li>At least 8 characters</li>
                          <li>One uppercase letter</li>
                          <li>One lowercase letter</li>
                          <li>One number</li>
                          <li>One special character (e.g., !@#$%^&*)</li>
                        </ul>
                      </div>
                    )}
                  </div>

                  <button
                    type="submit"
                    className="btn btn-primary w-100 py-2 mb-4 fw-semibold"
                  >
                    Sign Up
                  </button>

                  <p className="text-center mb-0">
                    Already have an account?{" "}
                    <a
                      href="/Account/SignIn"
                      className="text-primary text-decoration-none fw-semibold"
                    >
                      Sign In
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
