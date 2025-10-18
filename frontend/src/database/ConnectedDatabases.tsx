import { useEffect, useState } from "react";
import { Card, Button, Modal } from "react-bootstrap";
import { FiTrash2, FiDatabase } from "react-icons/fi";
import { useNavigate } from "react-router-dom";
import useAuth from "../hooks/useAuth";
import {
  getUserConnections,
  deleteConnection,
  type DatabaseConnection,
} from "../services/DatabaseService";

export default function ConnectedDatabases() {
  const navigate = useNavigate();
  const { userId, isAuthenticated, loading } = useAuth();

  const [connections, setConnections] = useState<DatabaseConnection[]>([]);
  const [selectedConnection, setSelectedConnection] =
    useState<DatabaseConnection | null>(null);
  const [showModal, setShowModal] = useState(false);

  const fetchConnections = async () => {
    if (!userId) return;
    try {
      const data = await getUserConnections(userId);
      setConnections(data);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    if (isAuthenticated) {
      fetchConnections();
    }
  }, [userId, isAuthenticated]);

  const handleCardClick = (connection: DatabaseConnection) => {
    setSelectedConnection(connection);
    setShowModal(true);
  };

  const handleDelete = async (connectionId: string) => {
    const confirmDelete = window.confirm(
      "Are you sure you want to delete this connection?"
    );
    if (!confirmDelete || !userId) return;

    try {
      await deleteConnection(connectionId, userId);
      setConnections((prev) => prev.filter((conn) => conn.id !== connectionId));
      if (selectedConnection?.id === connectionId) setShowModal(false);
    } catch (err) {
      console.error("Failed to delete connection:", err);
    }
  };

  if (loading) return <div>Loading...</div>;

  return (
    <div className="container my-4">
      <h3 className="mb-4">Your Database Connections</h3>

      {connections.length === 0 && (
        <p>No connections found. Connect a database to get started.</p>
      )}

      <div className="d-flex flex-wrap gap-3">
        {connections.map((conn) => (
          <Card
            key={conn.id}
            style={{ width: "250px", cursor: "pointer", position: "relative" }}
            className="shadow-sm"
          >
            <Card.Body onClick={() => handleCardClick(conn)}>
              <Card.Title className="d-flex justify-content-between align-items-center">
                <span>{conn.instanceName}</span>
                <FiTrash2
                  color="red"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDelete(conn.id);
                  }}
                  style={{ cursor: "pointer" }}
                />
              </Card.Title>
              <Card.Subtitle className="mb-2 text-muted">
                {conn.provider} - {conn.dbType}
              </Card.Subtitle>
              <Card.Text>
                DB Name: {conn.dbName}
                <br />
                User: {conn.dbUser}
              </Card.Text>
            </Card.Body>
          </Card>
        ))}
      </div>

      <div className="mt-4">
        <Button
          variant="primary"
          onClick={() => navigate("/Account/DatabaseConnection")}
        >
          <FiDatabase className="me-2" />
          Connect Database
        </Button>
      </div>

      {/* Modal */}
      <Modal show={showModal} onHide={() => setShowModal(false)} centered>
        <Modal.Header closeButton>
          <Modal.Title>Database Connection Details</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {selectedConnection && (
            <>
              <p>
                <strong>Provider:</strong> {selectedConnection.provider}
              </p>
              <p>
                <strong>Database Type:</strong> {selectedConnection.dbType}
              </p>
              <p>
                <strong>Instance Name:</strong>{" "}
                {selectedConnection.instanceName}
              </p>
              <p>
                <strong>Database Name:</strong> {selectedConnection.dbName}
              </p>
              <p>
                <strong>User:</strong> {selectedConnection.dbUser}
              </p>
              <p>
                <strong>Connected On:</strong>{" "}
                {new Date(selectedConnection.connectedOn).toLocaleString()}
              </p>
            </>
          )}
        </Modal.Body>
        <Modal.Footer>
          <Button
            variant="primary"
            onClick={() => {
              if (selectedConnection) {
                navigate(
                  `/database/databasedescription/${selectedConnection.id}`
                );
              }
            }}
          >
            See Table Descriptions
          </Button>
          <Button
            variant="danger"
            onClick={() =>
              selectedConnection && handleDelete(selectedConnection.id)
            }
          >
            Delete
          </Button>
          <Button variant="secondary" onClick={() => setShowModal(false)}>
            Close
          </Button>
        </Modal.Footer>
      </Modal>
    </div>
  );
}
