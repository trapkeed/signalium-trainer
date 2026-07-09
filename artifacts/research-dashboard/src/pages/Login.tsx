import { useState } from "react";
import { useAuth } from "@/lib/auth";
import { Activity, ShieldAlert, Lock, ArrowRight, Loader2 } from "lucide-react";

export function Login() {
  const { login, isVerifying, error } = useAuth();
  const [token, setToken] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!token.trim()) return;
    await login(token.trim());
  };

  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center p-4 relative overflow-hidden">
      {/* Background decoration */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[400px] bg-primary/5 blur-[120px] rounded-full pointer-events-none" />
      
      <div className="w-full max-w-sm space-y-8 relative z-10">
        <div className="flex flex-col items-center space-y-4 text-center">
          <div className="w-12 h-12 bg-card border border-border rounded-lg flex items-center justify-center shadow-lg relative">
            <div className="absolute inset-0 bg-primary/20 blur-xl rounded-full" />
            <Activity className="w-6 h-6 text-primary relative z-10" />
          </div>
          <div className="space-y-1">
            <h1 className="text-2xl font-bold tracking-tight text-foreground">Research Lab</h1>
            <p className="text-sm text-muted-foreground">
              Signalium Automated Trading Ops
            </p>
          </div>
        </div>

        <div className="bg-card border border-border rounded-xl p-6 shadow-2xl">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <label htmlFor="token" className="text-xs font-mono text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
                <Lock className="w-3 h-3" />
                Admin Token
              </label>
              <div className="relative">
                <input
                  id="token"
                  type="password"
                  value={token}
                  onChange={(e) => setToken(e.target.value)}
                  placeholder="••••••••••••••••"
                  className="w-full bg-input border border-border rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary font-mono text-foreground placeholder:text-muted-foreground transition-all"
                  autoComplete="off"
                  autoFocus
                />
              </div>
            </div>

            {error && (
              <div className="flex items-start gap-2 text-destructive text-sm bg-destructive/10 border border-destructive/20 p-3 rounded-md">
                <ShieldAlert className="w-4 h-4 shrink-0 mt-0.5" />
                <p>{error}</p>
              </div>
            )}

            <button
              type="submit"
              disabled={!token.trim() || isVerifying}
              className="w-full bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed h-9 rounded-md text-sm font-medium transition-colors flex items-center justify-center gap-2"
            >
              {isVerifying ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <>
                  Authenticate <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </form>
        </div>

        <p className="text-xs text-muted-foreground text-center font-mono">
          RESTRICTED ACCESS • AUTHORIZED PERSONNEL ONLY
        </p>
      </div>
    </div>
  );
}
