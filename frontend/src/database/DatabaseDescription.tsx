import { useEffect, useState } from "react";
import { useParams, useNavigate, useLocation } from "react-router-dom";
import DatabaseEditor from "./DatabaseEditor";
import { Container, Spinner } from "react-bootstrap";
import {
  getDatabaseTables,
  updateDatabaseTables,
  type Table,
} from "../services/DatabaseService";

// Component to display and edit database description
export default function DatabaseDescription() {
  const { connectionId } = useParams<{ connectionId: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const fromNewConnection = location.state?.fromNewConnection ?? false;

  // State to hold database tables and loading status
  const [tables, setTables] = useState<Table[]>([]);
  const [loading, setLoading] = useState(false);
  const [refetching, setRefetching] = useState(false);

  // Fetch tables (shared logic)
  const fetchTables = async (isRefetch = false) => {
    if(!connectionId) return;

    try {
      if (isRefetch) {
        setRefetching(true);
      } else {
        setLoading(true);
      }

      // Fetch data from API
      const data = await getDatabaseTables(connectionId!);
      setTables(data);
    } catch (error) {
      console.error("Failed to fetch tables:", error);
      alert("Failed to fetch database schema.");
    } finally {
      if (isRefetch) {
        setRefetching(false);
      } else {
        setLoading(false);
      }
    }
  };

  // Initial fetch on mount
  useEffect(() => {
    fetchTables(false);
  }, [connectionId]);

  // Handle saving updated tables
  const handleSave = async (updatedTables: Table[]) => {
    if (!connectionId) return;
    try {
      await updateDatabaseTables(connectionId, updatedTables);
      setTables(updatedTables);
      alert("Tables updated successfully!");

      // Redirect based on how user came here
      if (fromNewConnection) {
        navigate("/chatinterface");
      } else {
        navigate("/database/connecteddatabases");
      }
    } catch (err) {
      console.error(err);
      alert("Failed to update tables.");
    }
  };

  // Show loading spinner while fetching data
  if (loading)
    return (
      <Container className="my-5 text-center">
        <Spinner animation="border" />
        <p className="mt-2">Loading database schema...</p>
      </Container>
    );

  return (
    <Container className="my-4 position-relative">

      {/* Loading Spinner */}
      {refetching && (
        <div
          className="position-fixed top-0 start-0 w-100 h-100 bg-dark bg-opacity-50 d-flex flex-column justify-content-center align-items-center"
          style={{ zIndex: 9999 }}
        >
          <Spinner animation="border" role="status" />
          <div className="mt-2 text-light fs-6">Refreshing schema...</div>
        </div>
      )}

      {/* Database Editor */}
      <DatabaseEditor
        databaseName={`Database: ${connectionId}`}
        tables={tables}
        onSave={handleSave}
        onRefetch={() => fetchTables(true)}
        disableEditing={refetching}
      />
    </Container>
  );
}