import { useState, useCallback } from "react";
import LoadingScreen from "./components/LoadingScreen";
import CaptioningView from "./components/CaptioningView";
import WelcomeScreen from "./components/WelcomeScreen";
import type { AppState } from "./types";

export default function App() {
  const [appState, setAppState] = useState<AppState>("welcome");

  const handleStart = useCallback(() => {
    setAppState("loading");
  }, []);

  const handleLoadingComplete = useCallback(() => {
    setAppState("captioning");
  }, []);

  return (
    <div className="App relative min-h-screen overflow-hidden bg-[#FFFAEB]">
      <div className="absolute inset-0 bg-[#FFFAEB]" />

      {appState !== "captioning" && (
        <div className="absolute inset-0 bg-[#FFFAEB]/90 backdrop-blur-sm" />
      )}

      {appState === "welcome" && <WelcomeScreen onStart={handleStart} />}

      {appState === "loading" && (
        <LoadingScreen onComplete={handleLoadingComplete} />
      )}

      {appState === "captioning" && <CaptioningView />}
    </div>
  );
}
