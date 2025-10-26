import { Route, Routes } from "react-router-dom";
import SignIn from "./SignIn";
import SignUp from "./SignUp";

// Account component that handles routing for sign-in and sign-up pages
export default function Account() {
  return (
    <div className="flex-grow-1 d-flex flex-column">
      <Routes>
        <Route path="/signin" element={<SignIn />} />
        <Route path="/signup" element={<SignUp />} />
      </Routes>
    </div>
  );
}
