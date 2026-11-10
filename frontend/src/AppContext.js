import { createContext, useContext } from "react";

// What App() has - the state and handlers of every hook it is composed of,
// and its own - for the parts of the screen that are files of their own.
// A component reads what it needs:
//
//     const { processSummary, handleGrid } = useApp();
//
// The value is a new object on every render of App, so every reader
// re-renders with it. That is what happened before the split too (one
// component, one render); it is not a place to hang expensive children
// without memoising them.
export const AppContext = createContext(null);

export function useApp() {
  const app = useContext(AppContext);
  if (app === null) throw new Error("useApp() must be used below <App>");
  return app;
}
