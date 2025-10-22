import { useEffect, useRef } from "react";
import "./NewChatModal.css";

interface Props {
  show: boolean;
  initialTitle?: string;
  onCancel: () => void;
  onConfirm: (title: string) => void;
}

export default function NewChatModal({
  show,
  initialTitle = "",
  onCancel,
  onConfirm,
}: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (show) {
      const t = setTimeout(() => inputRef.current?.focus(), 80);
      return () => clearTimeout(t);
    }
  }, [show]);

  if (!show) return null;

  return (
    <>
      <div
        role="dialog"
        aria-modal="true"
        className="position-fixed top-0 start-0 w-100 h-100 d-flex align-items-center justify-content-center nc-overlay"
        onClick={onCancel}
      >
        <div
          className="bg-white nc-panel p-4"
          style={{ width: 520, maxWidth: "92%" }}
          onClick={(e) => e.stopPropagation()}
        >
          <h5 className="mb-3">Create new chat</h5>

          <div className="mb-3">
            <label
              htmlFor="new-chat-title"
              className="form-label small text-muted"
            >
              Title (optional)
            </label>
            <input
              id="new-chat-title"
              ref={inputRef}
              className="form-control nc-input"
              defaultValue={initialTitle}
              placeholder="Give your chat a title"
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  onConfirm((e.target as HTMLInputElement).value.trim());
                } else if (e.key === "Escape") {
                  onCancel();
                }
              }}
            />
          </div>

          <div className="d-flex justify-content-end gap-2">
            <button className="btn btn-outline-secondary" onClick={onCancel}>
              Cancel
            </button>
            <button
              className="btn btn-primary"
              onClick={() => {
                const value = inputRef.current?.value ?? "";
                onConfirm(value.trim());
              }}
            >
              Create
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
