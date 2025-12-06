import { createContext, useState, useEffect, type ReactNode } from "react";
import { getProfile } from "../services/authService";

interface UserProfile {
  userId: string;
  email: string;
  firstName: string;
  lastName: string;
  avatarUrl?: string;
}

interface AuthContextType {
  isAuthenticated: boolean;
  loading: boolean;
  userId: string;
  userData: UserProfile | null;
  login: (token: string, user: any) => void;
  logout: () => void;
}

export const AuthContext = createContext<AuthContextType>({
  isAuthenticated: false,
  loading: true,
  userId: "",
  userData: null,
  login: () => {},
  logout: () => {},
});

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);
  const [userId, setUserId] = useState<string>("");
  const [userData, setUserData] = useState<UserProfile | null>(null);

  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem("token");
      // If no token exists, stop loading and return
      if (!token) {
        setLoading(false);
        return;
      }

      try {
        const response = await getProfile();
        if (response.status === 200) {
          const profileData = response.data;
          setUserData(profileData);
          setUserId(profileData.user_id || profileData.id); 
          setIsAuthenticated(true);
        }
      } catch (error){
        console.error("Auth check failed", error);
        logout();
      } finally {
        setLoading(false);
      }
    };
    checkAuth();
  }, []);

  // Login Action
  const login = (token: string, user: any) => {
    localStorage.setItem("token", token);
    setIsAuthenticated(true);
    setUserData(user);
    setUserId(user.user_id);
  };

  // Logout Action
  const logout = () => {
    localStorage.removeItem("token");
    setIsAuthenticated(false);
    setUserData(null);
    setUserId("");
  };

  return (
    <AuthContext.Provider value={{ isAuthenticated, loading, userId, userData, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};