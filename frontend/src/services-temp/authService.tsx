import axios from "axios";

// Enable sending cookies with requests
axios.defaults.withCredentials = true;

// Base API URL for user-related endpoints
const API_URL = `${import.meta.env.VITE_BACKEND_URL}/user`;

// Authentication credentials type definition
export interface AuthCredentials {
  firstName?: string;
  lastName?: string;
  email: string;
  password: string;
}

// Register a new user
export const register = (credentials: AuthCredentials) => {
  return axios.post(`${API_URL}/signup`, credentials);
};

// Sign in existing user
export const signIn = (credentials: AuthCredentials) => {
  return axios.post(`${API_URL}/signin`, credentials);
};

// Get authenticated user's profile
export const getProfile = () => {
  return axios.get(`${API_URL}/profile`);
};

// Sign out user
export const signOut = () => {
  return axios.post(`${API_URL}/logout`, null);
};
