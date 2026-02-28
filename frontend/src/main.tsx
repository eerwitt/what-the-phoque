import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.tsx";
import "./index.css";
import { VLMProvider } from "./context/VLMContext.tsx";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <VLMProvider>
      <App />
    </VLMProvider>
  </React.StrictMode>,
);
