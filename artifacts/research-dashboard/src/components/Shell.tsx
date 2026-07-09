import { ReactNode } from "react";
import { Link, useLocation } from "wouter";
import { useAuth } from "@/lib/auth";
import { Activity, AlertTriangle, CheckSquare, GitMerge, LineChart, Target, Box, Database, LogOut } from "lucide-react";
import { useCurrentDeployment, useHealthStatus } from "@/hooks/use-api";

const NAV_ITEMS = [
  { path: "/", label: "Overview", icon: Activity },
  { path: "/symbols", label: "Symbols", icon: LineChart },
  { path: "/calibration", label: "Calibration", icon: Target },
  { path: "/drift", label: "Drift Monitor", icon: AlertTriangle },
  { path: "/promotions", label: "Promotions", icon: GitMerge },
  { path: "/lineage", label: "Lineage", icon: Database },
  { path: "/features", label: "Features", icon: Box },
  { path: "/approvals", label: "Approvals", icon: CheckSquare },
];

export function Shell({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  const { logout } = useAuth();
  
  const { data: deployment } = useCurrentDeployment();
  const { data: health } = useHealthStatus();

  const isCritical = Object.values(health?.status?.dimensions || {}).some(d => d.severity === "CRITICAL");
  const isWarning = Object.values(health?.status?.dimensions || {}).some(d => d.severity === "WARNING");

  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground font-sans">
      {/* Top Navigation Bar */}
      <header className="h-14 border-b border-border bg-card flex items-center justify-between px-4 shrink-0">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-primary rounded-sm shadow-[0_0_8px_rgba(0,229,155,0.5)]"></div>
            <span className="font-bold tracking-tight text-sm uppercase text-foreground">Signalium Lab</span>
          </div>
          
          <nav className="flex items-center gap-1">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const active = location === item.path;
              return (
                <Link
                  key={item.path}
                  href={item.path}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                    active 
                      ? "bg-primary/10 text-primary" 
                      : "text-muted-foreground hover:text-foreground hover:bg-secondary"
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>
        
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-3 pr-4 border-r border-border">
            <div className="flex flex-col">
              <span className="text-[10px] text-muted-foreground uppercase">Active Model</span>
              <span className="text-foreground">{deployment?.active?.model_version || "None"}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] text-muted-foreground uppercase">System Status</span>
              <div className="flex items-center gap-1.5">
                <div className={`w-2 h-2 rounded-full ${isCritical ? "bg-destructive" : isWarning ? "bg-accent" : "bg-primary"}`} />
                <span className={isCritical ? "text-destructive" : isWarning ? "text-accent" : "text-primary"}>
                  {isCritical ? "CRITICAL" : isWarning ? "WARNING" : "HEALTHY"}
                </span>
              </div>
            </div>
          </div>
          <button 
            onClick={logout}
            className="flex items-center gap-1.5 text-muted-foreground hover:text-foreground transition-colors uppercase tracking-wider"
          >
            <LogOut className="w-3.5 h-3.5" />
            Logout
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 overflow-auto p-4 md:p-6 bg-background">
        <div className="max-w-7xl mx-auto space-y-6">
          {children}
        </div>
      </main>
    </div>
  );
}
