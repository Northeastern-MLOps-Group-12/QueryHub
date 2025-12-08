import axios from "axios";
import api from "./api";

// Enable sending cookies with requests
axios.defaults.withCredentials = true;

// Authentication credentials type definition
export interface AuthCredentials {
  first_name?: string;
  last_name?: string;
  email: string;
  password: string;
}

// Register a new user
export const register = (credentials: AuthCredentials) => {
  return api.post(`/auth/signup`, credentials);
};

// Sign in existing user
export const signIn = (credentials: AuthCredentials) => {
  console.log("AuthService - signIn called with credentials:", credentials);
  return api.post(`/auth/signin`, credentials);
};

// Get authenticated user's profile
export const getProfile = () => {
  return api.get(`/auth/profile`);
};