import { 
  useSummary, 
  useCurrentDeployment, 
  useHealthStatus, 
  useHealthAlerts, 
  useApprovals 
} from "@/hooks/use-api";
import { Card, CardHeader, CardContent, LoadingScreen, ErrorScreen, EmptyScreen, Badge, ValueDisplay, formatTime } from "@/components/ui-utils";
import { Activity, AlertTriangle, ShieldCheck, CheckSquare, Zap, Clock, ShieldAlert } from "lucide-react";
import { Link } from "wouter";

export default function Overview() {
  const { data: deployment, isLoading: isLoadingDep, error: errorDep } = useCurrentDeployment();
  const { data: health, isLoading: isLoadingHealth, error: errorHealth } = useHealthStatus();
  const { data: alertsData, isLoading: isLoadingAlerts } = useHealthAlerts({ limit: 5 });
  const { data: approvalsData, isLoading: isLoadingApp } = useApprovals({ status: "PENDING", limit: 5 });

  if (isLoadingDep || isLoadingHealth) return <LoadingScreen />;
  if (errorDep) return <ErrorScreen error={errorDep} />;
  if (errorHealth) return <ErrorScreen error={errorHealth} />;

  const active = deployment?.active;
  const dimensions = Object.entries(health?.status?.dimensions || {});
  const isCritical = dimensions.some(([_, d]) => d.severity === "CRITICAL");
  const isWarning = dimensions.some(([_, d]) => d.severity === "WARNING");
  
  const alerts = alertsData?.alerts || [];
  const approvals = approvalsData?.approval_requests || [];

  return (
    <div className="space-y-6">
      {/* Top Banner: Global Status */}
      <div className={`p-6 rounded-lg border flex items-center gap-6 ${
        isCritical ? "bg-destructive/10 border-destructive/30" : 
        isWarning ? "bg-accent/10 border-accent/30" : 
        "bg-primary/10 border-primary/30"
      }`}>
        <div className={`p-4 rounded-full ${
          isCritical ? "bg-destructive/20 text-destructive" : 
          isWarning ? "bg-accent/20 text-accent" : 
          "bg-primary/20 text-primary"
        }`}>
          {isCritical ? <ShieldAlert className="w-8 h-8" /> : 
           isWarning ? <AlertTriangle className="w-8 h-8" /> : 
           <ShieldCheck className="w-8 h-8" />}
        </div>
        <div>
          <h2 className={`text-xl font-mono font-bold uppercase tracking-widest ${
            isCritical ? "text-destructive" : 
            isWarning ? "text-accent" : 
            "text-primary"
          }`}>
            {isCritical ? "System Critical" : isWarning ? "System Warning" : "System Healthy"}
          </h2>
          <p className="text-sm text-muted-foreground mt-1">
            {isCritical ? "Critical dimensions detected in health status. Immediate action required." :
             isWarning ? "Warnings active in model telemetry. Monitoring recommended." :
             "All model telemetry dimensions are within normal operating parameters."}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Active Deployment */}
        <Card className="col-span-1 md:col-span-2">
          <CardHeader 
            title="Active Model Deployment" 
            icon={<Zap className="w-4 h-4 text-primary" />}
            action={
              <Badge variant={active?.status === "ACTIVE" ? "success" : "default"}>
                {active?.status || "UNKNOWN"}
              </Badge>
            }
          />
          <CardContent className="grid grid-cols-2 sm:grid-cols-4 gap-6">
            {active ? (
              <>
                <ValueDisplay label="Model Version" value={active.model_version} />
                <ValueDisplay label="Deployed At" value={active.deployed_at ? formatTime(active.deployed_at) : "N/A"} />
                <ValueDisplay label="Weights Hash" value={
                  <span className="truncate max-w-[100px] block" title={active.weights_url}>
                    {active.weights_url?.split('/').pop() || "N/A"}
                  </span>
                } />
                <ValueDisplay label="Strategy" value={active.metadata?.strategy as string || "N/A"} />
              </>
            ) : (
              <div className="col-span-4 py-4"><EmptyScreen message="No active model deployed" /></div>
            )}
          </CardContent>
        </Card>

        {/* Dimension Health Overview */}
        <Card className="col-span-1">
          <CardHeader title="Health Dimensions" action={<Link href="/drift" className="text-xs text-primary hover:underline">View details</Link>} />
          <CardContent noPadding>
            <div className="divide-y divide-border">
              {dimensions.length > 0 ? dimensions.map(([name, dim]) => (
                <div key={name} className="p-3 flex items-center justify-between hover:bg-secondary/20 transition-colors">
                  <span className="text-sm font-mono text-foreground capitalize">{name}</span>
                  <Badge variant={
                    dim.severity === "CRITICAL" ? "destructive" :
                    dim.severity === "WARNING" ? "warning" : "success"
                  }>
                    {dim.severity}
                  </Badge>
                </div>
              )) : (
                <div className="p-4"><EmptyScreen message="No health data" /></div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Open Health Alerts */}
        <Card>
          <CardHeader 
            title="Recent Alerts" 
            subtitle="Current health anomalies"
            action={<Link href="/drift" className="text-xs text-primary hover:underline">All alerts</Link>}
          />
          <CardContent noPadding>
            {isLoadingAlerts ? <div className="p-4"><LoadingScreen /></div> : alerts.length > 0 ? (
              <div className="divide-y divide-border">
                {alerts.map(alert => (
                  <div key={alert.alert_id} className="p-4 flex flex-col gap-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Badge variant={alert.severity === "CRITICAL" ? "destructive" : "warning"}>{alert.severity}</Badge>
                        <span className="text-xs font-mono text-muted-foreground uppercase">{alert.dimension}</span>
                      </div>
                      <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                        <Clock className="w-3 h-3" /> {formatTime(alert.created_at)}
                      </span>
                    </div>
                    <p className="text-sm text-foreground">{alert.message || "No description provided."}</p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-8">
                <EmptyScreen message="No active health alerts" icon={ShieldCheck} />
              </div>
            )}
          </CardContent>
        </Card>

        {/* Pending Approvals */}
        <Card>
          <CardHeader 
            title="Action Queue" 
            subtitle="Approvals requiring review"
            action={<Link href="/approvals" className="text-xs text-primary hover:underline">Manage queue</Link>}
          />
          <CardContent noPadding>
            {isLoadingApp ? <div className="p-4"><LoadingScreen /></div> : approvals.length > 0 ? (
              <div className="divide-y divide-border">
                {approvals.map(req => (
                  <div key={req.request_id} className="p-4 flex flex-col gap-2 bg-accent/5">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="border-accent text-accent">{req.gate_type}</Badge>
                        <span className="text-xs font-mono text-muted-foreground uppercase">{req.resource_type || "Resource"}</span>
                      </div>
                      <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                        <Clock className="w-3 h-3" /> {formatTime(req.created_at)}
                      </span>
                    </div>
                    <p className="text-sm text-foreground">{req.action_summary || `Review requested for ${req.resource_id}`}</p>
                    <div className="flex justify-end mt-2">
                      <Link href="/approvals" className="text-xs font-mono uppercase tracking-wide bg-accent text-accent-foreground px-3 py-1.5 rounded hover:bg-accent/90 transition-colors">
                        Review Request
                      </Link>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-8">
                <EmptyScreen message="Approval queue clear" icon={CheckSquare} />
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
