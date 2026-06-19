import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";
import "./App.css";
import Overview from "./pages/Overview";
import Chat from "./pages/Chat";
import Agent from "./pages/Agent";
import Structured from "./pages/Structured";
import Memory from "./pages/Memory";
import Logging from "./pages/Logging";

const NAV: { to: string; label: string; end?: boolean }[] = [
  { to: "/", label: "Overview", end: true },
  { to: "/chat", label: "Streaming Chat" },
  { to: "/agent", label: "Agent & Tools" },
  { to: "/structured", label: "Structured Output" },
  { to: "/memory", label: "Memory" },
  { to: "/logging", label: "Logging & Traces" },
];

export default function App() {
  return (
    <BrowserRouter>
      <div className="shell">
        <aside className="nav">
          <div className="brand">
            ai-core <span>demo</span>
          </div>
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.end}
              className={({ isActive }) => "navlink" + (isActive ? " active" : "")}
            >
              {n.label}
            </NavLink>
          ))}
          <div className="footnote">
            React → axum → ai-core → Ollama
          </div>
        </aside>
        <main className="content">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/chat" element={<Chat />} />
            <Route path="/agent" element={<Agent />} />
            <Route path="/structured" element={<Structured />} />
            <Route path="/memory" element={<Memory />} />
            <Route path="/logging" element={<Logging />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
