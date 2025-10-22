import { useState } from "react";
import DatabaseEditor from "./DatabaseEditor"; // make sure the path is correct
import { Container } from "react-bootstrap";

interface Column {
  name: string;
  description: string;
}

interface Table {
  name: string;
  description: string;
  columns: Column[];
}

export default function DatabaseDescription() {
  // Initial database and tables structure
  const initialTables: Table[] = [
    {
      name: "Users",
      description: "Stores user information",
      columns: [
        { name: "id", description: "Primary key" },
        { name: "name", description: "User full name" },
        { name: "email", description: "User email address" },
      ],
    },
    {
      name: "Orders",
      description: "Stores orders made by users",
      columns: [
        { name: "id", description: "Primary key" },
        { name: "user_id", description: "ID of the user who made the order" },
        { name: "amount", description: "Total order amount" },
      ],
    },
    {
      name: "Products",
      description: "Stores product information",
      columns: [
        { name: "id", description: "Primary key" },
        { name: "name", description: "Product name" },
        { name: "price", description: "Product price" },
      ],
    },
  ];

  const [tables, setTables] = useState<Table[]>(initialTables);

  // Handle saving updates from DatabaseEditor
  const handleSave = (updatedTables: Table[]) => {
    setTables(updatedTables);
    console.log("Updated tables:", updatedTables);
    // TODO: make API call here to save to backend
  };

  return (
    <Container>
      <DatabaseEditor
        databaseName="MyDatabase"
        tables={tables}
        onSave={handleSave}
      />
    </Container>
  );
}
