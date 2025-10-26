import { useState } from "react";
import { Accordion, Card, Button, Form, Container } from "react-bootstrap";

// Column type definition
interface Column {
  name: string;
  description: string;
}

// Table type definition
interface Table {
  name: string;
  description: string;
  columns: Column[];
}

// Props for DatabaseEditor component
interface DatabaseProps {
  databaseName: string;
  tables: Table[];
  onSave: (tables: Table[]) => void;
}

// Component to edit database tables and their descriptions
export default function DatabaseEditor({
  databaseName,
  tables: initialTables,
  onSave,
}: DatabaseProps) {
  const [tables, setTables] = useState<Table[]>(initialTables);
  const [openTableIndexes, setOpenTableIndexes] = useState<number[]>([0]);
  const [editing, setEditing] = useState(false);
  const [originalTables, setOriginalTables] = useState<Table[]>(initialTables);

  // Toggle accordion item open/close
  const handleToggle = (index: number) => {
    setOpenTableIndexes((prevIndexes) =>
      prevIndexes.includes(index)
        ? prevIndexes.filter((i) => i !== index)
        : [...prevIndexes, index]
    );
  };

  // Handle changes to column descriptions
  const handleColumnChange = (
    tableIndex: number,
    columnIndex: number,
    value: string
  ) => {
    const updatedTables = [...tables];
    updatedTables[tableIndex].columns[columnIndex].description = value;
    setTables(updatedTables);
  };

  // Handle changes to table descriptions
  const handleTableDescriptionChange = (tableIndex: number, value: string) => {
    const updatedTables = [...tables];
    updatedTables[tableIndex].description = value;
    setTables(updatedTables);
  };

  // Toggle editing mode
  const toggleEditing = () => {
    if (!editing) {
      setOriginalTables(
        tables.map((t) => ({
          ...t,
          columns: t.columns.map((c) => ({ ...c })),
        }))
      );
      setEditing(true);
    } else {
      setTables(originalTables);
      setEditing(false);
    }
  };

  return (
    <Container className="my-4">
      {/* Header with database name and edit button */}
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h3 className="mb-0">Database Schema: {databaseName}</h3>
        <Button
          variant={editing ? "secondary" : "primary"}
          onClick={toggleEditing}
        >
          {editing ? "Cancel Editing" : "Edit Descriptions"}
        </Button>
      </div>

      {/* Accordion for tables and their columns */}
      <Accordion
        alwaysOpen
        activeKey={openTableIndexes.map((index) => index.toString())}
      >
        {tables.map((table, tIndex) => (
          <Card key={table.name} className="mb-3 shadow-sm">
            <Accordion.Item eventKey={tIndex.toString()}>

              {/* Accordion header and body for each table */}
              <Accordion.Header onClick={() => handleToggle(tIndex)}>
                {table.name}
              </Accordion.Header>

              {/* Accordion body with table and column descriptions */}
              <Accordion.Body>
                <Form.Group className="mb-3">
                  <Form.Label>Table Description</Form.Label>
                  <Form.Control
                    as="textarea"
                    rows={2}
                    value={table.description}
                    onChange={(e) =>
                      handleTableDescriptionChange(tIndex, e.target.value)
                    }
                    placeholder="Describe the purpose and content of this table"
                    disabled={!editing}
                  />
                </Form.Group>

                {/* Column descriptions */}
                <div>
                  <h5 className="fs-6 mb-3">Columns</h5>
                  {table.columns.map((col, cIndex) => (
                    <Form.Group
                      key={col.name}
                      className="d-flex align-items-center mb-2"
                    >
                      <div style={{ flex: 1, fontWeight: "500" }}>
                        {col.name}
                      </div>
                      <Form.Control
                        type="text"
                        style={{ flex: 2, marginLeft: "1rem" }}
                        value={col.description}
                        onChange={(e) =>
                          handleColumnChange(tIndex, cIndex, e.target.value)
                        }
                        placeholder="Describe the column"
                        disabled={!editing}
                      />
                    </Form.Group>
                  ))}
                </div>
              </Accordion.Body>
            </Accordion.Item>
          </Card>
        ))}
      </Accordion>

      {/* Save button */}
      {editing && (
        <div className="mt-3 text-end">
          <Button onClick={() => onSave(tables)} variant="primary">
            Save Descriptions
          </Button>
        </div>
      )}
    </Container>
  );
}
