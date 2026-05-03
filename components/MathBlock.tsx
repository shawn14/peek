"use client";
import { useState } from "react";

export function MathBlock({
  label = "Show the math",
  children,
}: {
  label?: string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="my-6">
      <button
        onClick={() => setOpen(!open)}
        className="text-sm text-zinc-600 hover:text-zinc-900 underline underline-offset-4"
      >
        {open ? `Hide` : label}
      </button>
      {open && (
        <div className="mt-4 p-4 rounded-lg bg-zinc-50 border border-zinc-200 text-sm text-zinc-700 font-mono leading-relaxed">
          {children}
        </div>
      )}
    </div>
  );
}
