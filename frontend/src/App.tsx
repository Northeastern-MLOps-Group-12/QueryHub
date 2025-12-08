import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import useAuth from "./hooks/useAuth";
import Navbar from "./Navbar";
import HomePage from "./home";
import Account from "./account";
import ChatInterface from "./chat-interface";
import ProtectedRoute from "./components/ProtectedRoute";
// import Footer from "./Footer";
import SignIn from "./account/SignIn";
import SignUp from "./account/SignUp";
import Database from "./database";

// Main Application Component
function App() {
  const { loading } = useAuth();
  if (loading) return <div>Loading...</div>;

  return (
    <div className="d-flex flex-column vh-100">
      <Router>
        {/* Navigation Bar */}
        <Navbar />
        <div
          className="d-flex flex-column flex-grow-1"
          style={{ paddingTop: "60px" }}
        >
          <Routes>
            <Route path="/" element={<HomePage />} />

            {/* SignIn page */}
            <Route path="/account/signin" element={<SignIn />} />

            {/* SignUp page */}
            <Route path="/account/signup" element={<SignUp />} />

            {/* Account pages */}
            <Route
              path="/account/*"
              element={<ProtectedRoute element={<Account />} />}
            />

            {/* Chat interface */}
            <Route
              path="/chatinterface"
              element={<ProtectedRoute element={<ChatInterface />} />}
            />

            {/* Database Routes */}
            <Route
              path="/database/*"
              element={<ProtectedRoute element={<Database />} />}
            />

            {/* Catch-all */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>

        {/* Footer */}
        {/* <Footer /> */}
      </Router>
    </div>
  );
}

export default App;
