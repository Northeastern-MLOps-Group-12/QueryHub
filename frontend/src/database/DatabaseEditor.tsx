import { useState, useEffect } from "react";
import {
  Accordion,
  Card,
  Button,
  Form,
  Container,
  Badge,
  Row,
  Col
} from "react-bootstrap";
import { FiRefreshCw, FiDatabase, FiTable, FiColumns } from "react-icons/fi";

//Types for Column
interface Column {
  name: string;
  dataType: string;
}

//Types for Table
interface Table {
  name: string;
  description: string;
  columns: Column[];
}

// Props for DatabaseEditor component
interface DatabaseProps {
  databaseName: string;
  databaseDescription: string;
  tables: Table[];
  onRefetch: () => void;
  disableEditing?: boolean;
}

// Main DatabaseEditor component
export default function DatabaseEditor({
  databaseName,
  databaseDescription,
  tables: externalTables,
  onRefetch,
  disableEditing = false,
}: DatabaseProps) {
  const [tables, setTables] = useState<Table[]>(externalTables);
  const [openTableIndexes, setOpenTableIndexes] = useState<string[]>(["0"]);

  // Sync external tables when they change
  useEffect(() => {
    setTables(externalTables);
  }, [externalTables]);

  return (
    <Container className="my-4 position-relative">
      <div className="mb-4 border-bottom pb-3">
        
        <div className="d-flex justify-content-between align-items-center mb-3">
          <h2 className="d-flex align-items-center gap-2 mb-0">
            <FiDatabase className="text-primary" />
            {databaseName}
          </h2>

          {/* <Button
            variant="outline-primary"
            onClick={onRefetch}
            disabled={disableEditing}
            className="d-flex align-items-center gap-2"
          >
            <FiRefreshCw
              className={disableEditing ? "spinner-border spinner-border-sm" : ""}
            />
            {disableEditing ? "Refreshing..." : "Re-fetch Schema"}
          </Button> */}
        </div>

        <div
          className="text-muted bg-light p-2 rounded border"
          style={{
            maxHeight: "8em",
            overflowY: "auto",
            whiteSpace: "pre-wrap",
            fontSize: "0.9rem",
          }}
        >
          {databaseDescription || "No database summary available."}
        </div>
      </div>

      {/* Tables Accordion */}
      <h5 className="mb-3 text-secondary">
        Tables <Badge bg="secondary">{tables.length}</Badge>
      </h5>
      {tables.length === 0 ? (
        <div className="text-center p-5 bg-light rounded text-muted">
          No tables found in this database.
        </div>
      ) : (
        <Accordion
          alwaysOpen
          activeKey={openTableIndexes}
          onSelect={(e) => setOpenTableIndexes(Array.isArray(e) ? e : [e || ""])}
        >
          {tables.map((table, tIndex) => (
            <Card key={table.name} className="mb-3 shadow-sm">
              <Accordion.Item eventKey={tIndex.toString()}>
                {/* Accordion Header and Body */}
                <Accordion.Header>
                  <div className="d-flex align-items-center gap-2">
                    <FiTable className="text-secondary" />
                    <strong>{table.name}</strong>
                    <span className="text-muted small ms-2">
                      ({table.columns.length} columns)
                    </span>
                  </div>
                </Accordion.Header>

                <Accordion.Body>
                  <Form.Group className="mb-4">
                    <Form.Label className="text-muted small fw-bold text-uppercase">
                      Table Description
                    </Form.Label>
                    <div className="p-2 bg-light rounded border">
                      {table.description || "No description available."}
                    </div>
                  </Form.Group>

                  {/* Column Groups */}
                  <div>
                    <h6 className="d-flex align-items-center gap-2 text-muted small fw-bold text-uppercase mb-2">
                      <FiColumns /> Columns
                    </h6>
                    <div className="border rounded overflow-hidden">
                      {/* Grid Header */}
                      <Row className="bg-secondary bg-opacity-10 py-2 px-2 m-0 border-bottom fw-bold text-secondary" style={{fontSize: "0.9rem"}}>
                        <Col xs={6}>Column Name</Col>
                        <Col xs={6}>Data Type</Col>
                      </Row>

                      {/* Grid Rows */}
                      {table.columns.length > 0 ? (
                        table.columns.map((col, idx) => (
                          <Row 
                            key={idx} 
                            className={`py-2 px-2 m-0 align-items-center ${idx !== table.columns.length - 1 ? "border-bottom" : ""}`}
                          >
                            <Col xs={6} className="fw-medium text-dark font-monospace">
                              {col.name}
                            </Col>
                            <Col xs={6} className="text-muted fst-italic small">
                              <Badge bg="light" text="dark" className="border">
                                {col.dataType}
                              </Badge>
                            </Col>
                          </Row>
                        ))
                      ) : (
                         <div className="p-3 text-muted fst-italic text-center">No columns found.</div>
                      )}
                    </div>
                  </div>
                </Accordion.Body>
              </Accordion.Item>
            </Card>
          ))}
        </Accordion>
      )}
    </Container>
  );
}