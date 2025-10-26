import { useState, useEffect } from "react";
import { getProfile } from "../services/AuthService";

// User profile type definition
interface UserProfile {
  userId: string;
  email: string;
  firstName: string;
  lastName: string;
  avatarUrl?: string;
}

// Custom hook to manage authentication state
const useAuth = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);
  const [userId, setUserId] = useState<string>("");
  const [userData, setUserData] = useState<UserProfile | null>(null);

  // Check authentication status on mount
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
