import axios from "axios";

axios.defaults.withCredentials = true;

const API_URL = `${import.meta.env.VITE_API_URL}/db`;

// Interface for database connection
export interface DatabaseConnection {
  id: string;
  provider: string;
  dbType: string;
  instanceName: string;
  dbName: string;
  dbUser: string;
  dbPassword?: string;
  createdAt: string;
}

// Payload for adding a database connection
export interface DatabaseConnectionPayload {
  provider: string;
  dbType: string;
  instanceName: string;
  dbName: string;
  dbUser: string;
  dbPassword: string;
}

// Add a new database connection
export const addDatabaseConnection = async (
  data: DatabaseConnectionPayload,
  userId: string
) => {
  return axios.post(`${API_URL}/connect`, data, {
    params: { userId },
  });
};

// Get all connections for current user
export const getUserConnections = async (userId: string) => {
  const res = await axios.get(`${API_URL}/getDbConnections`, {
    params: { userId },
  });

  return res.data as DatabaseConnection[];
};

// Delete a connection by ID
export const deleteConnection = async (
  connectionId: string,
  userId: string
) => {
  return axios.delete(`${API_URL}/deleteConnection/${connectionId}`, {
    params: { userId },
  });
};
