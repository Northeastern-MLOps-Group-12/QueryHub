import { Navigate, useLocation } from "react-router-dom";
import useAuth from "../hooks/useAuth";
import { type ReactElement } from "react";

interface ProtectedRouteProps {
  element: ReactElement;
}

const ProtectedRoute = ({ element }: ProtectedRouteProps) => {
  const { isAuthenticated, loading } = useAuth();
  const location = useLocation();

  // Debugging: Uncomment the line below to see exactly what useAuth is returning
  console.log("ProtectedRoute Check:", { isAuthenticated, loading, path: location.pathname });

  if (loading) return <div>Loading...</div>;

  // If authenticated, render the protected element
  if (isAuthenticated) {
    return element;
  }

  // If not authenticated, redirect to SignIn
  return <Navigate to="/account/signin" state={{ from: location }} replace />;
};

export default ProtectedRoute;