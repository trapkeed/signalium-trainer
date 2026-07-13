import { useState } from "react";
import { useExperiments, useTrainingRuns, useApproveExperiment, useRejectExperiment } from "@/hooks/use-api";
import { type Experiment } from "@/lib/api";
import { Card, CardHeader, CardContent, LoadingScreen, ErrorScreen, EmptyScreen, Badge, formatTime } from "@/components/ui-utils";
import { FlaskConical, ChevronDown, ChevronRight, Check, X, GitCommit, Database } from "lucide-react";
import * as Dialog from "@radix-ui/react-dialog";

function statusVariant(status: string): "success" | "warning" | "destructive" | "outline" | "default" {
  const s = status?.toUpperCase();
  if (s === "COMPLETED" || s === "READY" || s === "ACTIVE") return "success";
  if (s === "PENDING" || s === "RUNNING") return "warning";
  if (s === "FAILED" || s === "REJECTED") return "destructive";
  return "outline";
}

function ExperimentRow({ exp }: { exp: Experiment }) {
  const [expanded, setExpanded] = useState(false);
  const { data: runsData, isLoading: runsLoading } = useTrainingRuns(
    expanded ? { experiment_id: exp.experiment_id } : undefined
  );

  const experiment = exp;

  const runs = runsData?.training_runs ?? [];

  return (
    <>
      <tr
        className="border-b border-border hover:bg-secondary/20 transition-colors cursor-pointer"
        onClick={() => setExpanded((v) => !v)}
      >
        <td className="px-4 py-3">
          <div className="flex items-center gap-2">
            {expanded ? (
              <ChevronDown className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
            )}
            <span className="text-sm font-mono font-bold text-foreground">{experiment.experiment_id}</span>
          </div>
          {experiment.name && (
            <span className="text-xs text-muted-foreground ml-6 block">{experiment.name}</span>
          )}
        </td>
        <td className="px-4 py-3">
          <Badge variant={statusVariant(experiment.status)}>{experiment.status || "—"}</Badge>
        </td>
        <td className="px-4 py-3 text-xs font-mono text-muted-foreground">
          {experiment.researcher || "—"}
        </td>
        <td className="px-4 py-3 text-xs font-mono text-foreground">
          {experiment.qde_version || "—"}
        </td>
        <td className="px-4 py-3 text-xs font-mono text-muted-foreground">
          {formatTime(experiment.created_at)}
        </td>
      </tr>
      {expanded && (
        <tr className="border-b border-border bg-secondary/10">
          <td colSpan={5} className="px-6 py-4">
            {runsLoading ? (
              <p className="text-xs font-mono text-muted-foreground animate-pulse">Loading runs…</p>
            ) : runs.length === 0 ? (
              <p className="text-xs font-mono text-muted-foreground italic">No training runs recorded for this experiment.</p>
            ) : (
              <div className="space-y-2">
                {runs.map((run) => (
                  <div
                    key={run.run_id}
                    className="flex items-center gap-4 text-xs font-mono bg-background/60 rounded px-3 py-2 border border-border"
                  >
                    <span className="text-foreground font-medium w-32 truncate" title={run.run_id}>
                      {run.run_id}
                    </span>
                    {run.model_version && (
                      <span className="text-primary bg-primary/10 px-2 py-0.5 rounded">
                        {run.model_version}
                      </span>
                    )}
                    <span className="flex items-center gap-1 text-muted-foreground">
                      <Database className="w-3 h-3" />
                      {run.dataset_version || "—"}
                    </span>
                    <span className="flex items-center gap-1 text-muted-foreground">
                      <GitCommit className="w-3 h-3" />
                      <span className="text-accent truncate max-w-[80px]">{run.code_commit || "—"}</span>
                    </span>
                    <span className="text-muted-foreground ml-auto">{formatTime(run.created_at)}</span>
                  </div>
                ))}
              </div>
            )}
          </td>
        </tr>
      )}
    </>
  );
}

export default function ExperimentsLab() {
  const [statusFilter, setStatusFilter] = useState("");
  const { data, isLoading, error } = useExperiments(
    statusFilter ? { status: statusFilter } : undefined
  );

  const approveExp = useApproveExperiment();
  const rejectExp = useRejectExperiment();

  const [actionTarget, setActionTarget] = useState<{ id: string; kind: "approve" | "reject" } | null>(null);
  const [reason, setReason] = useState("");

  if (isLoading) return <LoadingScreen />;
  if (error) return <ErrorScreen error={error} />;

  const experiments = data?.experiments ?? [];

  const handleAction = (e: React.FormEvent) => {
    e.preventDefault();
    if (!actionTarget) return;
    if (actionTarget.kind === "approve") {
      approveExp.mutate({ experimentId: actionTarget.id, reason: reason || undefined });
    } else {
      rejectExp.mutate({ experimentId: actionTarget.id, reason });
    }
    setActionTarget(null);
    setReason("");
  };

  const STATUS_FILTERS = ["", "PENDING", "ACTIVE", "COMPLETED", "FAILED"];

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-mono font-bold uppercase tracking-tight">Experiments</h1>
          <p className="text-sm text-muted-foreground mt-1">Research Lab experiment registry</p>
        </div>
        <div className="flex gap-2">
          {STATUS_FILTERS.map((s) => (
            <button
              key={s || "ALL"}
              onClick={() => setStatusFilter(s)}
              className={`px-3 py-1.5 text-xs font-mono uppercase rounded transition-colors ${
                statusFilter === s
                  ? "bg-secondary text-foreground font-bold"
                  : "text-muted-foreground hover:bg-secondary/50"
              }`}
            >
              {s || "ALL"}
            </button>
          ))}
        </div>
      </div>

      <Card>
        <CardHeader
          title="Experiment Registry"
          icon={<FlaskConical className="w-4 h-4 text-primary" />}
          subtitle="Click a row to expand training runs"
        />
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-y border-border bg-secondary/30">
                {["Experiment ID", "Status", "Researcher", "QDE Version", "Created"].map((h) => (
                  <th key={h} className="px-4 py-3 text-xs font-mono text-muted-foreground uppercase tracking-wider">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {experiments.length > 0 ? (
                experiments.map((exp) => <ExperimentRow key={exp.experiment_id} exp={exp as never} />)
              ) : (
                <tr>
                  <td colSpan={5} className="p-8">
                    <EmptyScreen message="No experiments found" icon={FlaskConical} />
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Approve / Reject experiments that are ready */}
      {experiments.some((e) => e.status?.toUpperCase() === "COMPLETED") && (
        <Card>
          <CardHeader title="Promotion Actions" subtitle="Experiments ready for review" />
          <div className="divide-y divide-border">
            {experiments
              .filter((e) => e.status?.toUpperCase() === "COMPLETED")
              .map((exp) => (
                <div key={exp.experiment_id} className="flex items-center justify-between px-4 py-3">
                  <div>
                    <span className="text-sm font-mono text-foreground">{exp.experiment_id}</span>
                    {exp.name && <span className="text-xs text-muted-foreground ml-2">{exp.name}</span>}
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setActionTarget({ id: exp.experiment_id, kind: "approve" })}
                      className="flex items-center gap-1 px-3 py-1.5 text-xs font-mono uppercase bg-primary/10 text-primary border border-primary/30 rounded hover:bg-primary/20 transition-colors"
                    >
                      <Check className="w-3.5 h-3.5" /> Approve
                    </button>
                    <button
                      onClick={() => setActionTarget({ id: exp.experiment_id, kind: "reject" })}
                      className="flex items-center gap-1 px-3 py-1.5 text-xs font-mono uppercase bg-destructive/10 text-destructive border border-destructive/30 rounded hover:bg-destructive/20 transition-colors"
                    >
                      <X className="w-3.5 h-3.5" /> Reject
                    </button>
                  </div>
                </div>
              ))}
          </div>
        </Card>
      )}

      {/* Action Dialog */}
      <Dialog.Root open={!!actionTarget} onOpenChange={(open) => !open && setActionTarget(null)}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-background/80 backdrop-blur-sm" />
          <Dialog.Content className="fixed top-1/2 left-1/2 max-w-[480px] w-[90vw] translate-x-[-50%] translate-y-[-50%] rounded-lg bg-card border border-border p-6 shadow-2xl focus:outline-none flex flex-col gap-4">
            <Dialog.Title
              className={`text-lg font-mono font-bold uppercase flex items-center gap-2 ${
                actionTarget?.kind === "approve" ? "text-primary" : "text-destructive"
              }`}
            >
              {actionTarget?.kind === "approve" ? (
                <><Check className="w-5 h-5" /> Approve Experiment</>
              ) : (
                <><X className="w-5 h-5" /> Reject Experiment</>
              )}
            </Dialog.Title>
            <Dialog.Description className="text-xs font-mono text-muted-foreground">
              Experiment: <span className="text-foreground">{actionTarget?.id}</span>
            </Dialog.Description>
            <form onSubmit={handleAction} className="flex flex-col gap-4">
              <textarea
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                placeholder={actionTarget?.kind === "reject" ? "Reason for rejection (required)…" : "Approval notes (optional)…"}
                required={actionTarget?.kind === "reject"}
                className="w-full min-h-[80px] bg-input border border-border rounded p-3 text-sm font-mono text-foreground focus:outline-none resize-none"
                autoFocus
              />
              <div className="flex justify-end gap-3">
                <button type="button" onClick={() => setActionTarget(null)}
                  className="px-4 py-2 text-sm font-mono uppercase text-muted-foreground hover:bg-secondary rounded transition-colors">
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={actionTarget?.kind === "reject" && !reason.trim()}
                  className={`px-4 py-2 text-sm font-mono uppercase rounded transition-colors disabled:opacity-50 ${
                    actionTarget?.kind === "approve"
                      ? "bg-primary text-primary-foreground hover:bg-primary/90"
                      : "bg-destructive text-destructive-foreground hover:bg-destructive/90"
                  }`}
                >
                  Confirm
                </button>
              </div>
            </form>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
}
