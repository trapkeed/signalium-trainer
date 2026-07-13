import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Route, Switch, Router as WouterRouter } from 'wouter';
import { AuthProvider, useAuth } from '@/lib/auth';
import { Shell } from '@/components/Shell';
import { Login } from '@/pages/Login';
import Overview from '@/pages/Overview';
import Symbols from '@/pages/Symbols';
import Calibration from '@/pages/Calibration';
import Drift from '@/pages/Drift';
import Promotions from '@/pages/Promotions';
import Lineage from '@/pages/Lineage';
import Features from '@/pages/Features';
import Approvals from '@/pages/Approvals';
import ExperimentsLab from '@/pages/ExperimentsLab';
import ShadowTests from '@/pages/ShadowTests';
import DeploymentCandidates from '@/pages/DeploymentCandidates';
import ModelTree from '@/pages/ModelTree';

const queryClient = new QueryClient();

function ProtectedRoutes() {
  const { isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    return <Login />;
  }

  return (
    <Shell>
      <Switch>
        <Route path="/" component={Overview} />
        <Route path="/symbols" component={Symbols} />
        <Route path="/calibration" component={Calibration} />
        <Route path="/drift" component={Drift} />
        <Route path="/promotions" component={Promotions} />
        <Route path="/lineage" component={Lineage} />
        <Route path="/features" component={Features} />
        <Route path="/approvals" component={Approvals} />
        <Route path="/experiments-lab" component={ExperimentsLab} />
        <Route path="/shadow" component={ShadowTests} />
        <Route path="/candidates" component={DeploymentCandidates} />
        <Route path="/model-tree" component={ModelTree} />
        <Route>
          <div className="flex flex-col items-center justify-center h-[50vh] space-y-4">
            <h1 className="text-4xl font-bold text-muted-foreground font-mono">404</h1>
            <p className="text-muted-foreground">Telemetry not found.</p>
          </div>
        </Route>
      </Switch>
    </Shell>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}>
          <ProtectedRoutes />
        </WouterRouter>
      </AuthProvider>
    </QueryClientProvider>
  );
}

export default App;
