import { useState } from "react";
import { Form, Button } from "react-bootstrap";
import { addDatabaseConnection } from "../services/DatabaseService";
import { providerOptions, dbTypeOptions } from "../data/dbOptions";
import useAuth from "../hooks/useAuth";
import { useNavigate } from "react-router-dom"; // import useNavigate

export default function DatabaseConnection() {
  const { userId } = useAuth();
  const navigate = useNavigate(); // initialize navigate

  const [formData, setFormData] = useState({
    provider: "",
    dbType: "",
    instanceName: "",
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
    try {
      setLoading(true);
      const res = await addDatabaseConnection(formData, userId);

      // Assuming your backend returns the new connection ID in res.data.id
      const connectionId = res.data.id;

      setSuccess("Database connected successfully!");
      setError("");

      // Redirect to the database description page
      navigate(`/database/databasedescription/${connectionId}`, {
        state: { fromNewConnection: true },
      });
    } catch (err: any) {
      setError(err.response?.data || "Failed to connect database");
      setSuccess("");
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

        {/* Instance Name */}
        <Form.Group className="mb-3">
          <Form.Label>Database Instance Name</Form.Label>
          <Form.Control
            type="text"
            name="instanceName"
            value={formData.instanceName}
            onChange={handleChange}
            placeholder="Enter instance name"
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
