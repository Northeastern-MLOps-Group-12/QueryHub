import { useState } from "react";
import { Form, Button } from "react-bootstrap";
import { addDatabaseConnection } from "../services/DatabaseService";
import { providerOptions, dbTypeOptions } from "../data/dbOptions";
import useAuth from "../hooks/useAuth";
import { useNavigate } from "react-router-dom";

export default function DatabaseConnection() {
  const { userId } = useAuth();
  const navigate = useNavigate();

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

  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState("");
  const [error, setError] = useState("");

  const handleChange: React.ChangeEventHandler<any> = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

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
          user_id: "111",
          connection_name: formData.connectionName,
          db_host: formData.host,
          provider: formData.provider,
          db_type: formData.dbType,
          db_user: formData.dbUser,
          db_password: formData.dbPassword,
          db_name: formData.dbName,
        },
      };

      const res = await addDatabaseConnection(payload);
      console.log("Add Connection Response:", res);

      const connectionId = res.data.id;

      setSuccess("Database connected successfully!");

      // Redirect to description page
      navigate(`/database/databasedescription/${connectionId}`, {
        state: { fromNewConnection: true },
      });
    } catch (err: any) {
      console.error("Error adding connection:", err);
      setError(err.response?.data || "Failed to connect to database");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container my-4">
      <h3>Connect a Database</h3>
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

        {success && <p className="text-success">{success}</p>}
        {error && <p className="text-danger">{error}</p>}

        <Button type="submit" disabled={loading}>
          {loading ? "Connecting..." : "Connect Database"}
        </Button>
      </Form>
    </div>
  );
}