import { useState } from "react";
import { useShadowTests, useShadowTest } from "@/hooks/use-api";
import { Card, CardHeader, CardContent, LoadingScreen, ErrorScreen, EmptyScreen, Badge, formatTime } from "@/components/ui-utils";
import { Eye, ChevronDown, ChevronRight, TrendingUp, BarChart2 } from "lucide-react";

function statusVariant(status: string): "success" | "warning" | "destructive" | "outline" {
  const s = status?.toUpperCase();
  if (s === "CONCLUDED" || s === "COMPLETE" || s === "PASSED") return "success";
  if (s === "ACTIVE" || s === "RUNNING") return "warning";
  if (s === "FAILED" || s === "REJECTED") return "destructive";
  return "outline";
}

function ShadowTestRow({ test }: { test: { test_id: string; challenger_version: string; champion_version?: string; status: string; agreement_rate?: number; rolling_sharpe?: number; conclusion?: string; created_at: number; [key: string]: unknown } }) {
  const [expanded, setExpanded] = useState(false);
  const { data: detail, isLoading } = useShadowTest(expanded ? test.test_id : "");

  const detailTest = detail?.test as typeof test | undefined;

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
            <span className="text-sm font-mono font-bold text-foreground">{test.test_id}</span>
          </div>
        </td>
        <td className="px-4 py-3">
          <div className="flex flex-col gap-0.5 text-xs font-mono">
            <span className="text-primary">↑ {test.challenger_version || "—"}</span>
            <span className="text-muted-foreground">vs {test.champion_version || "champion"}</span>
          </div>
        </td>
        <td className="px-4 py-3">
          <Badge variant={statusVariant(test.status)}>{test.status || "—"}</Badge>
        </td>
        <td className="px-4 py-3 text-sm font-mono text-foreground">
          {test.agreement_rate !== undefined && test.agreement_rate !== null
            ? `${(test.agreement_rate * 100).toFixed(1)}%`
            : "—"}
        </td>
        <td className="px-4 py-3 text-sm font-mono text-foreground">
          {test.rolling_sharpe !== undefined && test.rolling_sharpe !== null
            ? test.rolling_sharpe.toFixed(3)
            : "—"}
        </td>
        <td className="px-4 py-3 text-xs font-mono text-muted-foreground">
          {formatTime(test.created_at)}
        </td>
      </tr>
      {expanded && (
        <tr className="border-b border-border bg-secondary/10">
          <td colSpan={6} className="px-6 py-4">
            {isLoading ? (
              <p className="text-xs font-mono text-muted-foreground animate-pulse">Loading detail…</p>
            ) : (
              <div className="space-y-3">
                {detailTest?.conclusion && (
                  <div className="bg-background/60 rounded p-3 border border-border">
                    <span className="text-[10px] font-mono text-muted-foreground uppercase block mb-1">Conclusion</span>
                    <p className="text-sm text-foreground">{detailTest.conclusion}</p>
                  </div>
                )}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  {[
                    ["Agreement Rate", test.agreement_rate !== undefined ? `${(test.agreement_rate * 100).toFixed(2)}%` : "N/A"],
                    ["Rolling Sharpe", test.rolling_sharpe !== undefined ? test.rolling_sharpe.toFixed(4) : "N/A"],
                    ["Challenger", test.challenger_version || "—"],
                    ["Champion", test.champion_version || "—"],
                  ].map(([label, value]) => (
                    <div key={label} className="bg-background/60 rounded p-3 border border-border">
                      <span className="text-[10px] font-mono text-muted-foreground uppercase block mb-1">{label}</span>
                      <span className="text-sm font-mono text-foreground">{value}</span>
                    </div>
                  ))}
                </div>
                {/* Extra fields from the detail response */}
                {detailTest && Object.entries(detailTest)
                  .filter(([k]) => !["test_id", "challenger_version", "champion_version", "status", "agreement_rate", "rolling_sharpe", "conclusion", "created_at"].includes(k))
                  .filter(([, v]) => typeof v !== "object" || v === null)
                  .slice(0, 8)
                  .map(([k, v]) => (
                    <div key={k} className="text-xs font-mono text-muted-foreground">
                      <span className="uppercase">{k}</span>:{" "}
                      <span className="text-foreground">{String(v)}</span>
                    </div>
                  ))
                }
              </div>
            )}
          </td>
        </tr>
      )}
    </>
  );
}

export default function ShadowTests() {
  const [statusFilter, setStatusFilter] = useState("");
  const { data, isLoading, error } = useShadowTests(statusFilter ? { status: statusFilter } : undefined);

  if (isLoading) return <LoadingScreen />;
  if (error) return <ErrorScreen error={error} />;

  const tests = data?.shadow_tests ?? [];
  const active = tests.filter((t) => ["ACTIVE", "RUNNING"].includes(t.status?.toUpperCase()));

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-mono font-bold uppercase tracking-tight">Shadow Tests</h1>
          <p className="text-sm text-muted-foreground mt-1">Challenger vs. champion model comparisons</p>
        </div>
        <div className="flex gap-2">
          {["", "ACTIVE", "CONCLUDED", "FAILED"].map((s) => (
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

      {/* Summary cards */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: "Total Tests", value: tests.length, icon: Eye },
          { label: "Active", value: active.length, icon: TrendingUp },
          {
            label: "Avg Agreement",
            value: tests.filter((t) => t.agreement_rate != null).length > 0
              ? `${(tests.reduce((s, t) => s + (t.agreement_rate ?? 0), 0) / tests.filter((t) => t.agreement_rate != null).length * 100).toFixed(1)}%`
              : "—",
            icon: BarChart2,
          },
        ].map(({ label, value, icon: Icon }) => (
          <Card key={label}>
            <CardContent className="flex items-center gap-4 py-4">
              <div className="p-2 bg-primary/10 rounded">
                <Icon className="w-5 h-5 text-primary" />
              </div>
              <div>
                <p className="text-[10px] font-mono text-muted-foreground uppercase">{label}</p>
                <p className="text-xl font-mono font-bold text-foreground">{value}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card>
        <CardHeader
          title="Shadow Test Registry"
          icon={<Eye className="w-4 h-4 text-primary" />}
          subtitle="Click a row to expand details"
        />
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-y border-border bg-secondary/30">
                {["Test ID", "Versions", "Status", "Agreement", "Sharpe", "Started"].map((h) => (
                  <th key={h} className="px-4 py-3 text-xs font-mono text-muted-foreground uppercase tracking-wider">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tests.length > 0 ? (
                tests.map((test) => (
                  <ShadowTestRow key={test.test_id} test={test} />
                ))
              ) : (
                <tr>
                  <td colSpan={6} className="p-8">
                    <EmptyScreen message="No shadow tests found" icon={Eye} />
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
