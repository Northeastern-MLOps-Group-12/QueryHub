import { Navigate } from "react-router-dom";
import UseAuth from "./UseAuth";

import { type ReactElement } from "react";

const ProtectedRoute = ({ element }: { element: ReactElement }) => {
  const { isAuthenticated, loading } = UseAuth();

  if (loading) {
    return <div>Loading...</div>;
  }

  return isAuthenticated ? element : <Navigate to="/Account/SignIn" />;
};

export default ProtectedRoute;
