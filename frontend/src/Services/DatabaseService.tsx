import axios from "axios";

axios.defaults.withCredentials = true;

const API_URL = `${import.meta.env.VITE_BACKEND_URL}/connect`;

// Interface for database connection
export interface DatabaseConnection {
  id: string;
  provider: string;
  dbType: string;
  connectionName: string;
  dbName: string;
  dbUser: string;
  dbPassword?: string;
  connectedOn: string;
}

// Payload for adding a database connection
export interface DatabaseConnectionPayload {
  engine: string;
  provider: string;
  config: {
    user_id: string;
    connection_name: string;
    db_host: string;
    provider: string;
    db_type: string;
    db_user: string;
    db_password: string;
    db_name: string;
  };
}

export interface Column {
  name: string;
  description: string;
}

export interface Table {
  name: string;
  description: string;
  columns: Column[];
}

// Add a new database connection
export const addDatabaseConnection = async (
  data: DatabaseConnectionPayload
) => {
  return axios.post(`${API_URL}/addConnection`, data);
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

// Fetch tables for a database connection
export const getDatabaseTables = async (connectionId: string) => {
  const res = await axios.get(`${API_URL}/${connectionId}`);
  return res.data as Table[];
};

// Update tables for a database connection
export const updateDatabaseTables = async (
  connectionId: string,
  tables: Table[]
) => {
  const res = await axios.put(`${API_URL}/${connectionId}`, { tables });
  return res.data;
};
