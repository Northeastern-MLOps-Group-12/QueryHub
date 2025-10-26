import { useEffect, useState } from "react";
import { Card, Button, Modal } from "react-bootstrap";
import { FiTrash2, FiDatabase } from "react-icons/fi";
import { useNavigate } from "react-router-dom";
import useAuth from "../hooks/useAuth";
import {
  getUserConnections,
  deleteConnection,
  type DatabaseConnection,
} from "../services/databaseService";

// Component to display and manage connected databases
export default function ConnectedDatabases() {
  const navigate = useNavigate();
  const { userId, isAuthenticated, loading } = useAuth();
  const [connections, setConnections] = useState<DatabaseConnection[]>([]);
  const [selectedConnection, setSelectedConnection] =
    useState<DatabaseConnection | null>(null);
  const [showModal, setShowModal] = useState(false);

  // Fetch user connections from the backend
  const fetchConnections = async () => {
    if (!userId) return;
    try {
      const data = await getUserConnections(userId);
      setConnections(data);
    } catch (err) {
      console.error(err);
    }
  };

  // Fetch connections on component mount and when userId changes
  useEffect(() => {
    if (isAuthenticated) {
      fetchConnections();
    }
  }, [userId, isAuthenticated]);

  // Handle card click to show modal with details
  const handleCardClick = (connection: DatabaseConnection) => {
    setSelectedConnection(connection);
    setShowModal(true);
  };

  // Handle deletion of a connection
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

      {/* Display a message if no connections are found */}
      {connections.length === 0 && (
        <p>No connections found. Connect a database to get started.</p>
      )}

      {/* Display connected database cards */}
      <div className="d-flex flex-wrap gap-3">
        {connections.map((conn) => (
          <Card
            key={conn.id}
            style={{ width: "250px", cursor: "pointer", position: "relative" }}
            className="shadow-sm"
          >
            <Card.Body onClick={() => handleCardClick(conn)}>

              {/* Card title with delete icon */}
              <Card.Title className="d-flex justify-content-between align-items-center">
                <span>{conn.connectionName}</span>
                <FiTrash2
                  color="red"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDelete(conn.id);
                  }}
                  style={{ cursor: "pointer" }}
                />
              </Card.Title>

              {/* Card subtitle and text */}
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

      {/* Button to navigate to database connection page */}
      <div className="mt-4">
        <Button
          variant="primary"
          onClick={() => navigate("/Account/DatabaseConnection")}
        >
          <FiDatabase className="me-2" />
          Connect Database
        </Button>
      </div>

      {/* Modal to show connection details */}
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
                <strong>Connection Name:</strong>{" "}
                {selectedConnection.connectionName}
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

        {/* Modal footer with action buttons */}
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

          {/* Delete button */}
          <Button
            variant="danger"
            onClick={() =>
              selectedConnection && handleDelete(selectedConnection.id)
            }
          >
            Delete
          </Button>

          {/* Close button */}
          <Button variant="secondary" onClick={() => setShowModal(false)}>
            Close
          </Button>
        </Modal.Footer>
      </Modal>
    </div>
  );
}
