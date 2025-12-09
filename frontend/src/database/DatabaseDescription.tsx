import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import DatabaseEditor from "./DatabaseEditor";
import { Container, Spinner, Alert} from "react-bootstrap";
import useAuth from "../hooks/useAuth";
import {
  getSingleDatabaseDetails,
  updateDatabaseConnection,
  type Table,
} from "../services/databaseService";

// Component to display and edit database description
export default function DatabaseDescription() {
  const { dbName } = useParams<{ dbName: string }>();
  const { userId } = useAuth();

  // State to hold database tables and loading status
  const [dbDescription, setDbDescription] = useState("");
  const [tables, setTables] = useState<Table[]>([]);
  const [loading, setLoading] = useState(false);
  const [refetching, setRefetching] = useState(false);
  const [error, setError] = useState("");

  // Fetch tables (shared logic)
  const fetchData = async (isRefetch: boolean) => {
    if (!dbName || !userId) return;
    setError("");

    try {
      if (isRefetch) {
        setRefetching(true);
      } else {
        setLoading(true);
      }

      let data;

      // Fetch data from API
      if (isRefetch) {
        await updateDatabaseConnection(Number(userId), dbName);
        data = await getSingleDatabaseDetails(userId, dbName);
      } else {
        data = await getSingleDatabaseDetails(userId, dbName);
      }

      if (data) {
        setTables(data.tables);
        setDbDescription(data.description);
      } else {
        setError("Database not found in response.");
      }
    } catch (error) {
      console.error("Failed to fetch tables:", error);
      setError("Failed to fetch database schema.");
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
    if (userId && dbName) {
      fetchData(false);
    }
  }, [userId, dbName]);

  // Show loading spinner while fetching data
  if (loading)
    return (
      <Container className="my-5 text-center">
        <Spinner animation="border" />
        <p className="mt-2">Loading database schema for {dbName}...</p>
      </Container>
    );

  return (
    <Container className="my-4 position-relative">

      {/* Loading Spinner */}
      {refetching && (
        <div
          className="position-fixed top-0 start-0 w-100 h-100 bg-dark bg-opacity-50 d-flex flex-column justify-content-center align-items-center"
          style={{ zIndex: 9999, backgroundColor: "rgba(0,0,0,0.5)" }}
        >
          <Spinner animation="border" variant="light" />
          <div className="mt-2 text-light fw-bold">Refetching Schema...</div>
        </div>
      )}

      {error && <Alert variant="danger">{error}</Alert>}

      {/* Database Editor */}
      <DatabaseEditor
        databaseName={dbName || ""}
        databaseDescription={dbDescription}
        tables={tables}
        onRefetch={() => fetchData(true)}
        disableEditing={refetching}
      />
    </Container>
  );
}