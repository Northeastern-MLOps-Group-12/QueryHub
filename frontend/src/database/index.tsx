import { Route, Routes } from "react-router-dom";
import ConnectedDatabases from "./ConnectedDatabases";
import DatabaseConnection from "./DatabaseConnection";
import DatabaseDescription from "./DatabaseDescription";

export default function Database() {
  return (
    <div className="flex-grow-1 d-flex flex-column">
      <Routes>
        <Route path="/connecteddatabases" element={<ConnectedDatabases />} />
        <Route path="/databaseconnections" element={<DatabaseConnection />} />
        <Route
          path="/databasedescription/*"
          element={<DatabaseDescription />}
        />
      </Routes>
    </div>
  );
}
