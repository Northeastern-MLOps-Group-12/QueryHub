import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import useAuth from "./Account/UseAuth";
import Navbar from "./Navbar";
import HomePage from "./Home";
import Account from "./Account";
import ChatInterface from "./ChatInterface";
import ProtectedRoute from "./Account/ProtectedRoute";
import Footer from "./Footer";
import SignIn from "./Account/SignIn";
import SignUp from "./Account/SignUp";
import DemoInterface from "./ChatInterface/DemoInterface";

function App() {
  const { loading } = useAuth();

  if (loading) return <div>Loading...</div>;

  return (
    <div className="d-flex flex-column vh-100">
      <Router>
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

            {/* Other account pages */}
            <Route
              path="/account/*"
              element={<ProtectedRoute element={<Account />} />}
            />

            {/* Chat interface */}
            <Route
              path="/chatinterface"
              element={<ProtectedRoute element={<ChatInterface />} />}
            />

            {/* Demo interface */}
            <Route
              path="/demointerface"
              element={<ProtectedRoute element={<DemoInterface />} />}
            />

            {/* Catch-all */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
        {/* <Footer /> */}
      </Router>
    </div>
  );
}

export default App;
