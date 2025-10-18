import { Navigate, useLocation } from "react-router-dom";
import useAuth from "../hooks/useAuth";
import { type ReactElement } from "react";

interface ProtectedRouteProps {
  element: ReactElement;
  publicRoutes?: string[];
}

const ProtectedRoute = ({
  element,
  publicRoutes = ["/", "/account/signin", "/account/signup"],
}: ProtectedRouteProps) => {
  const { isAuthenticated, loading } = useAuth();
  const location = useLocation();

  if (loading) return <div>Loading...</div>;

  // If route is public, allow it
  if (
    publicRoutes.some(
      (route) => route.toLowerCase() === location.pathname.toLowerCase()
    )
  ) {
    return element;
  }

  // If logged in, allow the route
  if (isAuthenticated) return element;

  // Not logged in → redirect to SignIn
  return <Navigate to="/account/signin" state={{ from: location }} replace />;
};

export default ProtectedRoute;
