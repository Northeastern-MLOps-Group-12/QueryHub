import { useState, useEffect } from "react";
import {
  Accordion,
  Card,
  Button,
  Form,
  Container,
  OverlayTrigger,
  Tooltip,
} from "react-bootstrap";

//Types for Column
interface Column {
  name: string;
  description: string;
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
  tables: Table[];
  onSave: (tables: Table[]) => void;
  onRefetch: () => void;
  disableEditing?: boolean;
}

// Main DatabaseEditor component
export default function DatabaseEditor({
  databaseName,
  tables: externalTables,
  onSave,
  onRefetch,
  disableEditing = false,
}: DatabaseProps) {
  const [tables, setTables] = useState<Table[]>(externalTables);
  const [openTableIndexes, setOpenTableIndexes] = useState<number[]>([0]);
  const [editing, setEditing] = useState(false);
  const [originalTables, setOriginalTables] = useState<Table[]>(externalTables);

  // Sync external tables when they change
  useEffect(() => {
    setTables(externalTables);
    setEditing(false);
  }, [externalTables]);

  // Toggle accordion item
  const handleToggle = (index: number) => {
    setOpenTableIndexes((prev) =>
      prev.includes(index) ? prev.filter((i) => i !== index) : [...prev, index]
    );
  };

  // Handle column description change
  const handleColumnChange = (
    tableIndex: number,
    columnIndex: number,
    value: string
  ) => {
    const updated = [...tables];
    updated[tableIndex].columns[columnIndex].description = value;
    setTables(updated);
  };

  // Handle table description change
  const handleTableDescriptionChange = (tableIndex: number, value: string) => {
    const updated = [...tables];
    updated[tableIndex].description = value;
    setTables(updated);
  };

  // Toggle editing mode
  const toggleEditing = () => {
    if (!editing) {
      setOriginalTables(JSON.parse(JSON.stringify(tables)));
      setEditing(true);
    } else {
      setTables(originalTables);
      setEditing(false);
    }
  };

  // Handle refetch click
  const handleRefetchClick = async () => {
    if (editing) return;
    setEditing(false);
    onRefetch();
  };

  return (
    <Container className="my-4 position-relative">
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h3 className="mb-0">{databaseName}</h3>
        <div className="d-flex gap-3">

          {/* Edit Button */}
          <Button
            variant={editing ? "secondary" : "primary"}
            onClick={toggleEditing}
            disabled={disableEditing}
          >
            {editing ? "Cancel Editing" : "Edit Descriptions"}
          </Button>

          {/* Refetch Button with Tooltip */}
          {editing ? (
            <OverlayTrigger
              placement="bottom"
              overlay={
                <Tooltip id="refetch-tooltip">
                  You are in editing mode. Re-fetch is disabled until you exit editing.
                </Tooltip>
              }
            >
              <span className="d-inline-block">
                <Button
                  variant="outline-primary"
                  disabled
                  style={{ pointerEvents: "none", opacity: 0.7 }}
                >
                  Re-fetch Schema
                </Button>
              </span>
            </OverlayTrigger>
          ) : (
            <Button
              variant="outline-primary"
              onClick={handleRefetchClick}
              disabled={disableEditing}
            >
              Re-fetch Schema
            </Button>
          )}
        </div>
      </div>

      {/* Tables Accordion */}
      <Accordion
        alwaysOpen
        activeKey={openTableIndexes.map((i) => i.toString())}
      >
        {tables.map((table, tIndex) => (
          <Card key={table.name} className="mb-3 shadow-sm">
            <Accordion.Item eventKey={tIndex.toString()}>

              {/* Accordion Header and Body */}
              <Accordion.Header onClick={() => handleToggle(tIndex)}>
                {table.name}
              </Accordion.Header>
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
                    placeholder="Describe this table"
                    disabled={!editing || disableEditing}
                  />
                </Form.Group>

                <h5 className="fs-6 mb-3">Columns</h5>

                {/* Column Groups */}
                {table.columns.map((col, cIndex) => (
                  <Form.Group
                    key={col.name}
                    className="d-flex align-items-center mb-2"
                  >
                    <div style={{ flex: 1, fontWeight: "500" }}>{col.name}</div>
                    <Form.Control
                      type="text"
                      style={{ flex: 2, marginLeft: "1rem" }}
                      value={col.description}
                      onChange={(e) =>
                        handleColumnChange(tIndex, cIndex, e.target.value)
                      }
                      placeholder="Describe this column"
                      disabled={!editing || disableEditing}
                    />
                  </Form.Group>
                ))}
              </Accordion.Body>
            </Accordion.Item>
          </Card>
        ))}
      </Accordion>

      {/* Save Button */}
      {editing && (
        <div className="mt-3 text-end">
          <Button
            onClick={() => onSave(tables)}
            variant="primary"
            disabled={disableEditing}
          >
            Save Descriptions
          </Button>
        </div>
      )}
    </Container>
  );
}