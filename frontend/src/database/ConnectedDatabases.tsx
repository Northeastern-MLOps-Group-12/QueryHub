import { useEffect, useState } from "react";
import { Row, Col, Card, Button, Spinner, Badge } from "react-bootstrap";
import {
  FiTrash2,
  FiDatabase,
  FiServer,
  FiArrowRight,
} from "react-icons/fi";
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
  const { userId, isAuthenticated, loading: authLoading } = useAuth();
  const [connections, setConnections] = useState<DatabaseConnection[]>([]);
  const [fetchingConnections, setFetchingConnections] = useState(true);

  // Fetch user connections from the backend
  const fetchConnections = async () => {
    if (!userId) return;
    try {
      setFetchingConnections(true);
      const data = await getUserConnections(userId);
      console.log("Fetched connections:", data);
      setConnections(data);
    } catch (err) {
      console.error(err);
    } finally {
      setFetchingConnections(false);
    }
  };

  // Fetch connections on component mount and when userId changes
  useEffect(() => {
    if (isAuthenticated) {
      fetchConnections();
    }
  }, [userId, isAuthenticated]);

  // Handle deletion of a connection
  const handleDelete = async (dbName: string) => {
    const confirmDelete = window.confirm(
      "Are you sure you want to delete this connection?"
    );
    if (!confirmDelete || !userId) return;

    try {
      console.log("Deleting connection:", dbName);
      console.log("For user ID:", userId);
      await deleteConnection(dbName, Number(userId));
      setConnections((prev) => prev.filter((conn) => conn.dbName !== dbName));
    } catch (err) {
      console.error("Failed to delete connection:", err);
    }
  };

  if (authLoading || fetchingConnections) {
    return (
      <div className="d-flex justify-content-center align-items-center vh-100">
        <Spinner animation="border" variant="primary" />
      </div>
    );
  }

  return (
    <div className="container my-4">
      <h2 className="mb-4 d-flex justify-content-center align-items-center">
        Your Database Connections
      </h2>

      {connections.length === 0 ? (
        <div className="text-center py-5 border rounded bg-light">
          <FiDatabase size={48} className="text-muted mb-3" />
          <h5>No connections found</h5>
          <p className="text-muted mb-4">
            Connect a database to generate analytics and chat with your data.
          </p>
          <Button
            variant="primary"
            onClick={() => navigate("/database/databaseconnection")}
          >
            <FiDatabase className="me-2" />
            Connect Database
          </Button>
        </div>
      ) : (
        <>
          <Row className="g-4">
            {" "}
            {/* g-4 adds consistent spacing between columns */}
            {connections.map((conn) => (
              <Col key={conn.dbName} xs={12} md={6} lg={4} xl={3}>
                <Card
                  // REMOVED fixed width. The Col controls the width now.
                  style={{ transition: "transform 0.2s" }}
                  className="shadow-sm border-2 h-100" // h-100 ensures all cards in a row are same height
                >
                  <Card.Body>
                    <div className="d-flex justify-content-between align-items-start">
                      {/* Icon and Name */}
                      <div className="d-flex align-items-center gap-2 overflow-hidden">
                        <div className="bg-light p-2 rounded text-primary">
                          <FiServer size={30} />
                        </div>
                        <div style={{ minWidth: 0 }}>
                          <h4
                            className="mb-0 text-truncate"
                            title={conn.dbName}
                          >
                            {conn.dbName}
                          </h4>
                          <Badge
                            bg="success"
                            className="fw-normal"
                            style={{ fontSize: "0.70rem" }}
                          >
                            Active
                          </Badge>
                        </div>
                      </div>

                      {/* --- ACTION ICONS (Refresh & Delete) --- */}
                      <div className="d-flex gap-1">
                        {/* 1. Refresh Schema Icon */}
                        {/* <Button
                          variant="link"
                          className="p-1 text-secondary opacity-75 hover-opacity-100"
                          title="Refresh Schema"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleRefreshSchema(conn.dbName);
                          }}
                        >
                          <FiRefreshCw size={20} />
                        </Button> */}

                        {/* 2. Delete Icon */}
                        {/* <Button
                          variant="link"
                          className="p-1 text-danger opacity-75 hover-opacity-100"
                          title="Delete Connection"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDelete(conn.dbName);
                          }}
                        >
                          <FiTrash2 size={20} />
                        </Button> */}
                      </div>
                    </div>
                  </Card.Body>

                  <Card.Footer className="bg-white border-top-0 pt-0 pb-3">
                    <Button
                      variant="outline-primary"
                      className="w-100 d-flex justify-content-between align-items-center"
                      onClick={() =>
                        navigate(`/database/databasedescription/${conn.dbName}`)
                      }
                    >
                      View Schema <FiArrowRight />
                    </Button>
                  </Card.Footer>
                </Card>
              </Col>
            ))}
          </Row>

          {/* Button to navigate to database connection page */}
          <div className="mt-4">
            <Button
              variant="primary"
              onClick={() => navigate("/database/databaseconnection")}
            >
              <FiDatabase className="me-2" />
              Connect New Database
            </Button>
          </div>
        </>
      )}
    </div>
  );
}
