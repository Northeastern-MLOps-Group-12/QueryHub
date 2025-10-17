import { useState, useEffect } from "react";
import { getProfile } from "../Services/AuthService";

const useAuth = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);
  const [userId, setUserId] = useState<string>("");

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const response = await getProfile();
        if (response.status === 200) {
          setUserId(response.data.userId);
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

  return { isAuthenticated, loading, userId };
};

export default useAuth;
