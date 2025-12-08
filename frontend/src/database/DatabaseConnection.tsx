import { useState } from "react";
import { Form, Button, Spinner } from "react-bootstrap";
import { addDatabaseConnection } from "../services/databaseService";
import { providerOptions, dbTypeOptions } from "../data/dbOptions";
import useAuth from "../hooks/useAuth";
import { useNavigate } from "react-router-dom";

// Component for connecting to a database
export default function DatabaseConnection() {
  const { userId } = useAuth();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState("");
  const [error, setError] = useState("");

  // Form state
  const [formData, setFormData] = useState({
    engine: "",
    provider: "",
    dbType: "",
    host: "",
    connectionName: "",
    dbName: "",
    dbUser: "",
    dbPassword: "",
  });

  // Handle form input changes
  const handleChange: React.ChangeEventHandler<any> = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setSuccess("");

    try {
      // Construct payload in backend-expected format
      const payload = {
        engine: formData.engine,
        provider: formData.provider,
        config: {
          user_id: String(userId),
          connection_name: formData.connectionName,
          db_host: formData.host,
          provider: formData.provider,
          db_type: formData.dbType,
          db_user: formData.dbUser,
          db_password: formData.dbPassword,
          db_name: formData.dbName,
        },
      };

      // Call service to add database connection
      await addDatabaseConnection(payload);
      setSuccess("Database connected successfully!");

      // Redirect to databases list
      navigate("/database/connecteddatabases");
    } catch (err: any) {
      console.error("Error adding connection:", err);

      // 1. Try to get the specific message from the object
      let message = "Failed to connect to database.";
      
      if (err.response?.data) {
          // Check if it's the specific "detail" field (FastAPI standard)
          if (typeof err.response.data.detail === "string") {
              message = err.response.data.detail;
          } 
          // Check if it's a "message" field
          else if (typeof err.response.data.message === "string") {
              message = err.response.data.message;
          }
          // Check if the data itself is just a string
          else if (typeof err.response.data === "string") {
              message = err.response.data;
          }
      }

      // 2. Clean up the message (Optional: remove technical SQL jargon)
      // Removes things like "(psycopg2.OperationalError)"
      const cleanMessage = message.replace(/\(psycopg2\.[^)]+\)\s*/, "");

      // 3. Set the state (GUARANTEED to be a string now)
      setError(cleanMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container my-5 position-relative">
      {loading && (
        <div
          className="position-fixed top-0 start-0 w-100 h-100 d-flex flex-column justify-content-center align-items-center"
          style={{
            zIndex: 9999,
            backgroundColor: "rgba(0,0,0,0.5)",
            backdropFilter: "blur(2px)",
          }}
        >
          <Spinner
            animation="border"
            variant="light"
            style={{ width: "3rem", height: "3rem" }}
          />
          <div className="mt-3 text-light fw-bold fs-5">
            Establishing Connection...
          </div>
          <div className="text-light small mt-1">
            This may take a few minutes
          </div>
        </div>
      )}
      <div className="row justify-content-center">
        <div className="col-lg-6 col-md-8">
          <div className="card shadow rounded-4 border-1">
            <div className="card-body p-4">
              <h3 className="mb-4 text-center fw-bold">Connect a Database</h3>
              <Form onSubmit={handleSubmit}>
                {/* Provider */}
                <Form.Group className="mb-3">
                  <Form.Label>DB Provider</Form.Label>
                  <Form.Select
                    name="provider"
                    value={formData.provider}
                    onChange={handleChange}
                    required
                  >
                    <option value="">Select Provider</option>
                    {providerOptions.map((p) => (
                      <option key={p} value={p}>
                        {p}
                      </option>
                    ))}
                  </Form.Select>
                </Form.Group>

                {/* Engine */}
                <Form.Group className="mb-3">
                  <Form.Label>DB Engine</Form.Label>
                  <Form.Select
                    name="engine"
                    value={formData.engine}
                    onChange={handleChange}
                    required
                    disabled={!formData.provider}
                  >
                    <option value="">Select Engine</option>
                    {formData.provider &&
                      dbTypeOptions[formData.provider].map((p) => (
                        <option key={p} value={p}>
                          {p}
                        </option>
                      ))}
                  </Form.Select>
                </Form.Group>

                {/* Database Type */}
                <Form.Group className="mb-3">
                  <Form.Label>Database Type</Form.Label>
                  <Form.Select
                    name="dbType"
                    value={formData.dbType}
                    onChange={handleChange}
                    required
                    disabled={!formData.provider}
                  >
                    <option value="">Select Database Type</option>
                    {formData.provider &&
                      dbTypeOptions[formData.provider].map((type) => (
                        <option key={type} value={type}>
                          {type}
                        </option>
                      ))}
                  </Form.Select>
                </Form.Group>

                {/* Host */}
                <Form.Group className="mb-3">
                  <Form.Label>Database Host</Form.Label>
                  <Form.Control
                    type="text"
                    name="host"
                    value={formData.host}
                    onChange={handleChange}
                    placeholder="Enter database host"
                    required
                  />
                </Form.Group>

                {/* Connection Name */}
                <Form.Group className="mb-3">
                  <Form.Label>Database Connection Name</Form.Label>
                  <Form.Control
                    type="text"
                    name="connectionName"
                    value={formData.connectionName}
                    onChange={handleChange}
                    placeholder="Enter connection name"
                    required
                  />
                </Form.Group>

                {/* Database Name */}
                <Form.Group className="mb-3">
                  <Form.Label>Database Name</Form.Label>
                  <Form.Control
                    type="text"
                    name="dbName"
                    value={formData.dbName}
                    onChange={handleChange}
                    placeholder="Enter database name"
                    required
                  />
                </Form.Group>

                {/* DB User */}
                <Form.Group className="mb-3">
                  <Form.Label>Database User</Form.Label>
                  <Form.Control
                    type="text"
                    name="dbUser"
                    value={formData.dbUser}
                    onChange={handleChange}
                    placeholder="Enter username"
                    required
                  />
                </Form.Group>

                {/* DB Password */}
                <Form.Group className="mb-3">
                  <Form.Label>Database Password</Form.Label>
                  <Form.Control
                    type="password"
                    name="dbPassword"
                    value={formData.dbPassword}
                    onChange={handleChange}
                    placeholder="Enter password"
                    required
                  />
                </Form.Group>

                {/* Success and Error Messages */}
                {success && <p className="text-success">{success}</p>}
                {error && (
                  <p className="text-danger text-center">
                    {typeof error === 'string' ? error : "An unexpected error occurred"}
                  </p>
                )}

                {/* Submit Button */}
                <Button type="submit" disabled={loading}>
                  {loading ? "Connecting..." : "Connect Database"}
                </Button>
              </Form>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
