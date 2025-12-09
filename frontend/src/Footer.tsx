// Footer component
export default function Footer() {
  return (
    <footer className="py-3 bg-white border-top text-center text-muted">
      <small>
        &copy; {new Date().getFullYear()} QueryHub. All Rights Reserved.
      </small>
    </footer>
  );
}
