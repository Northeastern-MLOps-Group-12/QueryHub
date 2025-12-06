import axios from "axios";
import api from "./api";

// Enable sending cookies with requests
axios.defaults.withCredentials = true;

export interface DatabaseConnection {
  dbName: string;
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

// Column type definition
export interface Column {
  name: string;
  dataType: string;
}

// Table type definition
export interface Table {
  name: string;
  description: string;
  columns: Column[];
}

// Database details including tables
export interface DatabaseDetails {
  dbName: string;
  description: string;
  tables: Table[];
}

// Transform Backend JSON to Frontend Interface
const transformResponse = (
  responseData: any,
  targetDbName: string
): DatabaseDetails | null => {
  if (!responseData || !responseData.connections) return null;

  const dbData = responseData.connections[targetDbName];

  if (!dbData) return null;

  const tablesArray: Table[] = Object.entries(dbData.tables || {}).map(
    ([tableName, tableData]: [string, any]) => ({
      name: tableName,
      description: tableData.description || "",
      columns: Array.isArray(tableData.columns)
        ? tableData.columns.map((col: any) => ({
            name: typeof col === "string" ? col : col.name,
            dataType: col.type || col.data_type || col.dtype || "Unknown",
          }))
        : [],
    })
  );

  return {
    dbName: targetDbName,
    description: dbData.dataset_summary || "",
    tables: tablesArray,
  };
};

// Add a new database connection
export const addDatabaseConnection = async (
  data: DatabaseConnectionPayload
) => {
  return api.post(`/connector/connect/addConnection`, data);
};

// Get all connections for current user
export const getUserConnections = async (userId: string) => {
  const res = await api.get(`/connector/connect/getAllConnections/${userId}`);

  const responseData = res.data;
  if (responseData && responseData.connections) {
    const transformedArray = Object.keys(responseData.connections).map(
      (key) => ({
        dbName: key,
      })
    );

    return transformedArray;
  }

  return [];
};

// Get details of a single database connection
export const getSingleDatabaseDetails = async (
  userId: string,
  dbName: string
) => {
  const res = await api.get(`/connector/connect/getAllConnections/${userId}`);
  return transformResponse(res.data, dbName);
};

// Update a database connection
export const updateDatabaseConnection = async (
  userId: number,
  dbName: string
) => {
  const res = await api.put(
    `/connector/connect/updateConnection/${userId}/${dbName}`
  );
  console.log("Update response data:", res.data);
  return transformResponse(res.data, dbName);
};

// Delete a connection by ID
export const deleteConnection = async (
  dbName: string,
  userId: number
) => {
  console.log("Deleting connection in service:", dbName, "for user ID:", userId);
  console.log("User ID type in service:", typeof userId);
  return api.delete(`/connector/connect/deleteConnection/${userId}/${dbName}`);
};