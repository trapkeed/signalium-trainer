import { ReactNode } from "react";
import { Loader2, AlertTriangle, CheckCircle2, Info } from "lucide-react";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div className={`bg-card border border-border rounded-lg overflow-hidden ${className}`}>
      {children}
    </div>
  );
}

export function CardHeader({ title, subtitle, action, icon, className = "" }: { title: ReactNode; subtitle?: ReactNode; action?: ReactNode; icon?: ReactNode; className?: string }) {
  return (
    <div className={`p-4 border-b border-border flex items-center justify-between gap-4 ${className}`}>
      <div className="flex items-center gap-3">
        {icon && <div className="shrink-0">{icon}</div>}
        <div className="space-y-1">
          <h3 className="font-mono text-sm font-semibold tracking-tight text-foreground uppercase">{title}</h3>
          {subtitle && <p className="text-xs text-muted-foreground">{subtitle}</p>}
        </div>
      </div>
      {action && <div>{action}</div>}
    </div>
  );
}

export function CardContent({ children, className = "", noPadding = false }: { children: ReactNode; className?: string; noPadding?: boolean }) {
  return (
    <div className={`${noPadding ? "" : "p-4"} ${className}`}>
      {children}
    </div>
  );
}

export function Badge({ children, variant = "default", className = "" }: { children: ReactNode; variant?: "default" | "success" | "warning" | "destructive" | "outline"; className?: string }) {
  const variants = {
    default: "bg-secondary text-secondary-foreground",
    success: "bg-primary/20 text-primary border border-primary/30",
    warning: "bg-accent/20 text-accent border border-accent/30",
    destructive: "bg-destructive/20 text-destructive border border-destructive/30",
    outline: "border border-border text-muted-foreground",
  };
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono font-medium uppercase tracking-wider ${variants[variant]} ${className}`}>
      {children}
    </span>
  );
}

export function LoadingScreen() {
  return (
    <div className="flex flex-col items-center justify-center h-64 space-y-3">
      <Loader2 className="w-6 h-6 animate-spin text-primary" />
      <p className="text-xs font-mono text-muted-foreground uppercase tracking-wider">Loading Telemetry...</p>
    </div>
  );
}

export function ErrorScreen({ error }: { error?: Error | string | unknown }) {
  const message = error instanceof Error ? error.message : typeof error === "string" ? error : "Failed to load data";
  return (
    <div className="flex flex-col items-center justify-center h-64 space-y-3 text-destructive p-4 border border-destructive/20 bg-destructive/5 rounded-lg">
      <AlertTriangle className="w-8 h-8" />
      <p className="text-sm font-mono uppercase tracking-wider font-semibold">System Error</p>
      <p className="text-xs text-destructive/80">{message}</p>
    </div>
  );
}

export function EmptyScreen({ message = "No data available", icon = Info }: { message?: string; icon?: any }) {
  const Icon = icon;
  return (
    <div className="flex flex-col items-center justify-center h-48 space-y-3 p-4 border border-dashed border-border rounded-lg bg-secondary/20">
      <Icon className="w-6 h-6 text-muted-foreground" />
      <p className="text-xs font-mono text-muted-foreground uppercase tracking-wider">{message}</p>
    </div>
  );
}

export function ValueDisplay({ label, value, subValue, trend }: { label: string; value: ReactNode; subValue?: ReactNode; trend?: "up" | "down" | "neutral" }) {
  return (
    <div className="flex flex-col">
      <span className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider mb-1">{label}</span>
      <div className="flex items-baseline gap-2">
        <span className="text-lg font-mono font-medium text-foreground">{value}</span>
        {subValue && <span className="text-xs font-mono text-muted-foreground">{subValue}</span>}
      </div>
    </div>
  );
}

export function formatTime(unixSeconds: number) {
  return new Date(unixSeconds * 1000).toLocaleString(undefined, { 
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false
  });
}
