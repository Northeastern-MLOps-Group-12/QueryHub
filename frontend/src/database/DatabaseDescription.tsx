import { useEffect, useState } from "react";
import { useParams, useNavigate, useLocation } from "react-router-dom";
import DatabaseEditor from "./DatabaseEditor";
import { Container } from "react-bootstrap";
import {
  getDatabaseTables,
  updateDatabaseTables,
  type Table,
} from "../services/DatabaseService";

export default function DatabaseDescription() {
  const { connectionId } = useParams<{ connectionId: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const fromNewConnection = location.state?.fromNewConnection ?? false;

  const [tables, setTables] = useState<Table[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (connectionId) {
      setLoading(true);
      getDatabaseTables(connectionId)
        .then((data) => setTables(data))
        .finally(() => setLoading(false));
    }
  }, [connectionId]);

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

  if (loading)
    return <Container className="my-4">Loading database schema...</Container>;

  return (
    <Container className="my-4">
      <DatabaseEditor
        databaseName={`Database: ${connectionId}`}
        tables={tables}
        onSave={handleSave}
      />
    </Container>
  );
}
