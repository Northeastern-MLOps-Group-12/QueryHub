export default function Footer() {
  return (
    <footer className="bg-dark text-white py-3">
      <div className=" text-center">
        <p className="mb-0">
          &copy; {new Date().getFullYear()} QueryHub <br />
          All rights reserved.
        </p>
      </div>
    </footer>
  );
}
