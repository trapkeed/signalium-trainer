import { useState } from "react";
import { useExperiments, useTrainingRuns } from "@/hooks/use-api";
import { Card, CardHeader, CardContent, LoadingScreen, ErrorScreen, EmptyScreen, Badge, formatTime } from "@/components/ui-utils";
import { Network, ChevronDown, ChevronRight, GitCommit, Database, FileBox } from "lucide-react";

function statusVariant(status: string): "success" | "warning" | "destructive" | "outline" {
  const s = status?.toUpperCase();
  if (s === "COMPLETED" || s === "READY") return "success";
  if (s === "ACTIVE" || s === "RUNNING" || s === "PENDING") return "warning";
  if (s === "FAILED" || s === "REJECTED") return "destructive";
  return "outline";
}

function ExperimentNode({ exp }: {
  exp: { experiment_id: string; name?: string; status: string; researcher?: string; qde_version?: string; created_at: number; [key: string]: unknown }
}) {
  const [open, setOpen] = useState(false);
  const { data: runsData, isLoading } = useTrainingRuns(
    open ? { experiment_id: exp.experiment_id } : undefined
  );
  const runs = runsData?.training_runs ?? [];

  return (
    <div className="border border-border rounded-lg overflow-hidden">
      {/* Experiment header row */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-3 px-4 py-3 bg-card hover:bg-secondary/30 transition-colors text-left"
      >
        {open ? (
          <ChevronDown className="w-4 h-4 text-muted-foreground shrink-0" />
        ) : (
          <ChevronRight className="w-4 h-4 text-muted-foreground shrink-0" />
        )}
        <div className="flex items-center gap-3 flex-1 min-w-0">
          <div className="flex flex-col min-w-0">
            <span className="text-sm font-mono font-bold text-foreground truncate">
              {exp.experiment_id}
            </span>
            {exp.name && (
              <span className="text-xs text-muted-foreground truncate">{exp.name}</span>
            )}
          </div>
          <Badge variant={statusVariant(exp.status)}>{exp.status || "—"}</Badge>
          {exp.qde_version && (
            <span className="text-xs font-mono text-muted-foreground hidden sm:inline">
              v{exp.qde_version}
            </span>
          )}
        </div>
        <div className="text-xs font-mono text-muted-foreground shrink-0 text-right">
          {exp.researcher && <span className="block">{exp.researcher}</span>}
          <span>{formatTime(exp.created_at)}</span>
        </div>
      </button>

      {/* Training runs sub-tree */}
      {open && (
        <div className="border-t border-border bg-secondary/10 px-4 py-3 space-y-2">
          {isLoading ? (
            <p className="text-xs font-mono text-muted-foreground animate-pulse py-2">
              Loading training runs…
            </p>
          ) : runs.length === 0 ? (
            <p className="text-xs font-mono text-muted-foreground italic py-2">
              No training runs for this experiment.
            </p>
          ) : (
            runs.map((run) => (
              <div
                key={run.run_id}
                className="ml-6 border border-border rounded bg-card px-3 py-2 flex flex-col sm:flex-row sm:items-center gap-3"
              >
                {/* Run ID + model version */}
                <div className="flex flex-col min-w-0 sm:w-44">
                  <span className="text-xs font-mono font-bold text-foreground truncate">
                    {run.run_id}
                  </span>
                  {run.model_version ? (
                    <span className="text-[10px] font-mono text-primary bg-primary/10 px-1.5 py-0.5 rounded mt-1 inline-block w-fit">
                      {run.model_version}
                    </span>
                  ) : (
                    <span className="text-[10px] font-mono text-muted-foreground italic mt-1">
                      no artifact
                    </span>
                  )}
                </div>

                {/* Hashes */}
                <div className="flex flex-col gap-1 flex-1 text-xs font-mono min-w-0">
                  <span className="flex items-center gap-1.5 text-muted-foreground">
                    <Database className="w-3 h-3 shrink-0" />
                    <span className="truncate">{run.dataset_version || "—"}</span>
                  </span>
                  <span className="flex items-center gap-1.5 text-muted-foreground">
                    <FileBox className="w-3 h-3 shrink-0" />
                    <span className="truncate">{run.feature_version || "—"}</span>
                  </span>
                  <span className="flex items-center gap-1.5 text-muted-foreground">
                    <GitCommit className="w-3 h-3 shrink-0" />
                    <span className="truncate text-accent">{run.code_commit || "—"}</span>
                  </span>
                </div>

                <span className="text-[10px] font-mono text-muted-foreground shrink-0">
                  {formatTime(run.created_at)}
                </span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}

export default function ModelTree() {
  const { data: expData, isLoading: expLoading, error: expError } = useExperiments();

  if (expLoading) return <LoadingScreen />;
  if (expError) return <ErrorScreen error={expError} />;

  const experiments = expData?.experiments ?? [];

  const counts = experiments.reduce(
    (acc, e) => {
      const s = e.status?.toUpperCase();
      if (s === "COMPLETED" || s === "READY") acc.completed++;
      else if (s === "ACTIVE" || s === "RUNNING") acc.active++;
      return acc;
    },
    { completed: 0, active: 0 }
  );

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-mono font-bold uppercase tracking-tight">Model Genealogy</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Experiment → training run → artifact lineage tree
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs font-mono">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-primary inline-block" />
            <span className="text-muted-foreground">{counts.completed} completed</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-accent inline-block" />
            <span className="text-muted-foreground">{counts.active} active</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-muted-foreground/40 inline-block" />
            <span className="text-muted-foreground">
              {experiments.length - counts.completed - counts.active} other
            </span>
          </span>
        </div>
      </div>

      <Card>
        <CardHeader
          title="Experiment Tree"
          icon={<Network className="w-4 h-4 text-primary" />}
          subtitle="Click an experiment to expand its training runs"
        />
        <CardContent noPadding>
          {experiments.length > 0 ? (
            <div className="p-4 space-y-2">
              {experiments.map((exp) => (
                <ExperimentNode key={exp.experiment_id} exp={exp} />
              ))}
            </div>
          ) : (
            <div className="p-8">
              <EmptyScreen
                message="No experiments in the registry yet"
                icon={Network}
              />
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
