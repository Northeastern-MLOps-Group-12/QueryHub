import axios from "axios";

axios.defaults.withCredentials = true;

const API_URL = `${import.meta.env.VITE_API_URL}/user`;

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
