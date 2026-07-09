import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { getAuthToken, onAuthChange, setAuthToken, verifyToken } from "@/lib/api";

interface AuthContextValue {
  isAuthenticated: boolean;
  isVerifying: boolean;
  error: string | null;
  login: (token: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(() => !!getAuthToken());
  const [isVerifying, setIsVerifying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Any 401/403 anywhere in the app clears the token (see api.ts); reflect that
  // immediately in auth state so protected routes hand off back to the login gate
  // instead of continuing to render with a dead session.
  useEffect(() => {
    return onAuthChange((token) => {
      setIsAuthenticated(!!token);
    });
  }, []);

  const login = async (token: string) => {
    setIsVerifying(true);
    setError(null);
    try {
      const ok = await verifyToken(token);
      if (ok) {
        setIsAuthenticated(true);
      } else {
        setError("Invalid admin token. Check the token and try again.");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not reach the Research Lab API.");
    } finally {
      setIsVerifying(false);
    }
  };

  const logout = () => {
    setAuthToken(null);
    setIsAuthenticated(false);
  };

  return (
    <AuthContext.Provider value={{ isAuthenticated, isVerifying, error, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
