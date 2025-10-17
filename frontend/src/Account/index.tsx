import { Route, Routes } from "react-router-dom";
import SignIn from "./SignIn";
import SignUp from "./SignUp";
import ConnectedDatabases from "./ConnectedDatabases";
import DatabaseConnection from "./DatabaseConnection";

export default function Account() {
  return (
    <div className="flex-grow-1 d-flex flex-column">
      <Routes>
        <Route path="/signin" element={<SignIn />} />
        <Route path="/signup" element={<SignUp />} />
        <Route path="/connecteddatabases" element={<ConnectedDatabases />} />
        <Route path="/databaseconnection" element={<DatabaseConnection />} />
      </Routes>
    </div>
  );
}
