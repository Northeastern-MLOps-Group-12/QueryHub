import { useState, useEffect } from "react";
import { getProfile } from "../Services/authService";

const useAuth = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);
  const [userEmail, setUserEmail] = useState<string>("");

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const response = await getProfile();
        if (response.status === 200) {
          setUserEmail(response.data.email);
          setIsAuthenticated(true);
        }
      } catch {
        setIsAuthenticated(false);
      } finally {
        setLoading(false);
      }
    };
    checkAuth();
  }, []);

  return { isAuthenticated, loading, userEmail };
};

export default useAuth;
