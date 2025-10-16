import {
  Navigate,
  Route,
  BrowserRouter as Router,
  Routes,
} from "react-router-dom";
import useAuth from "./Account/UseAuth";
import Navbar from "./Navbar";
import HomePage from "./Home";
import Account from "./Account";
import ProtectedRoute from "./Account/ProtectedRoute";
import ChatInterface from "./ChatInterface";
import Footer from "./Footer";

function App() {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return <div>Loading...</div>;
  }

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
            <Route
              path="/Account/*"
              element={
                isAuthenticated ? (
                  <Navigate to="/ChatInterface" replace />
                ) : (
                  <Account />
                )
              }
            />
            <Route
              path="/ChatInterface"
              element={<ProtectedRoute element={<ChatInterface />} />}
            />
          </Routes>
        </div>
        <Footer />
      </Router>
    </div>
  );
}

export default App;
