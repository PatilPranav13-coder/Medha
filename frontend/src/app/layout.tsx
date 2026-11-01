import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Medha — Technical Intelligence Workspace",
  description: "A focused workspace for coding, research, architecture, and engineering.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
