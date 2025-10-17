import { useState, useEffect } from "react";
import { getProfile } from "../Services/AuthService";

interface UserProfile {
  userId: string;
  email: string;
  firstName: string;
  lastName: string;
  avatarUrl?: string;
}

const useAuth = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);
  const [userId, setUserId] = useState<string>("");
  const [userData, setUserData] = useState<UserProfile | null>(null);

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const response = await getProfile();
        if (response.status === 200) {
          setUserData(response.data);
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

  return { isAuthenticated, loading, userId, userData };
};

export default useAuth;
