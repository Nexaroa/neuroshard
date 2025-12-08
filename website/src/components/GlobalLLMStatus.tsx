import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { 
  Activity, 
  TrendingUp, 
  TrendingDown,
  Minus,
  Users, 
  Zap, 
  CheckCircle2, 
  XCircle,
  AlertTriangle,
  Database,
  RefreshCw,
  Server,
  Layers,
  Clock
} from 'lucide-react';
import axios from 'axios';
import { API_URL } from '../config/api';
import { SEO } from './SEO';

// Format loss values for better readability
// Small values like 0.0001 become "1.0×10⁻⁴" or "0.01%"
const formatLoss = (loss: number | null | undefined): string => {
  if (loss === null || loss === undefined || loss <= 0 || isNaN(loss) || !isFinite(loss)) return '—';
  
  // For values >= 0.01, show normally
  if (loss >= 0.01) {
    return loss.toFixed(4);
  }
  
  // For smaller values, use scientific notation with superscript
  const exponent = Math.floor(Math.log10(loss));
  const mantissa = loss / Math.pow(10, exponent);
  
  // Unicode superscript digits: ⁰¹²³⁴⁵⁶⁷⁸⁹⁻
  const superscriptMap: { [key: string]: string } = {
    '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
    '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻'
  };
  
  const expStr = exponent.toString().split('').map(c => superscriptMap[c] || c).join('');
  return `${mantissa.toFixed(1)}×10${expStr}`;
};

interface TrainingNode {
  node_id: string;
  training_rounds: number;
  current_loss: number | null;
}

interface TrainingStatus {
  is_training: boolean;
  training_verified: boolean;
  is_converging: boolean;
  global_loss: number;
  loss_trend: string;
  hash_agreement_rate: number;
  total_nodes_training: number;
  total_training_steps: number;
  total_tokens_trained: number;
  data_shards_covered: number;
  sync_success_rate: number;
  diloco: {
    enabled: boolean;
    inner_steps_config: number;
    outer_steps_completed: number;
  };
  nodes: TrainingNode[];
  source: string;
}

// Status badge - consistent with other pages
const StatusBadge = ({ type, label }: { type: 'success' | 'warning' | 'error'; label: string }) => {
  const styles = {
    success: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    warning: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    error: 'bg-red-500/10 text-red-400 border-red-500/20'
  };
  
  const icons = {
    success: <CheckCircle2 className="w-3.5 h-3.5" />,
    warning: <AlertTriangle className="w-3.5 h-3.5" />,
    error: <XCircle className="w-3.5 h-3.5" />
  };

  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${styles[type]}`}>
      {icons[type]}
      {label}
    </span>
  );
};

// Stat card - simple and clean
const StatCard = ({ 
  icon: Icon, 
  label, 
  value, 
  subtitle,
  trend
}: {
  icon: React.ElementType;
  label: string;
  value: string | number;
  subtitle?: string;
  trend?: 'up' | 'down' | 'stable';
}) => (
  <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5">
    <div className="flex items-center justify-between mb-3">
      <div className="p-2 bg-slate-800 rounded-lg">
        <Icon className="w-5 h-5 text-cyan-400" />
      </div>
      {trend && (
        <span className={`${
          trend === 'up' ? 'text-emerald-400' : 
          trend === 'down' ? 'text-red-400' : 'text-slate-400'
        }`}>
          {trend === 'up' && <TrendingUp className="w-4 h-4" />}
          {trend === 'down' && <TrendingDown className="w-4 h-4" />}
          {trend === 'stable' && <Minus className="w-4 h-4" />}
        </span>
      )}
    </div>
    <p className="text-sm text-slate-400 mb-1">{label}</p>
    <p className="text-2xl font-semibold text-white">{value}</p>
    {subtitle && <p className="text-xs text-slate-500 mt-1">{subtitle}</p>}
  </div>
);

export const GlobalLLMStatus = () => {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const fetchStatus = useCallback(async () => {
    try {
      const response = await axios.get(`${API_URL}/api/training/global`);
      setStatus(response.data);
      setLastUpdate(new Date());
      setError(null);
    } catch (err) {
      console.error('Failed to fetch training status:', err);
      setError('Unable to connect to training network');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 10000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const handleRefresh = () => {
    setRefreshing(true);
    fetchStatus();
  };

  const getTrendFromLossTrend = (trend: string): 'up' | 'down' | 'stable' => {
    if (trend === 'improving' || trend === 'improving_strongly') return 'up';
    if (trend === 'degrading' || trend === 'needs_improvement') return 'down';
    return 'stable';
  };

  const getStatusType = (verified: boolean): 'success' | 'warning' | 'error' => {
    if (verified) return 'success';
    return 'warning';
  };

  // Loading state - simple spinner consistent with other pages
  if (loading) {
    return (
      <div className="min-h-screen bg-slate-950 pt-28 pb-12 px-6">
        <div className="container mx-auto max-w-7xl">
          <div className="flex items-center justify-center py-24">
            <RefreshCw className="w-6 h-6 text-cyan-400 animate-spin" />
            <span className="ml-3 text-slate-400">Loading training status...</span>
          </div>
        </div>
      </div>
    );
  }

  // Error state
  if (error || !status) {
    return (
      <div className="min-h-screen bg-slate-950 pt-28 pb-12 px-6">
        <div className="container mx-auto max-w-7xl">
          <div className="mb-8">
            <h1 className="text-4xl font-bold text-white mb-2">Training Status</h1>
            <p className="text-slate-400">Real-time decentralized training progress</p>
          </div>
          <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-8 text-center">
            <AlertTriangle className="w-10 h-10 text-amber-400 mx-auto mb-3" />
            <p className="text-white font-medium mb-2">Connection Error</p>
            <p className="text-slate-400 text-sm mb-4">{error || 'Unable to fetch training data'}</p>
            <button
              onClick={handleRefresh}
              className="inline-flex items-center gap-2 px-4 py-2 bg-cyan-500 hover:bg-cyan-400 text-white rounded-lg text-sm font-medium transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  const hashAgreementPercent = Math.round((status.hash_agreement_rate || 0) * 100);
  const stepsUntilSync = status.diloco?.enabled 
    ? (status.diloco.inner_steps_config || 500) - ((status.total_training_steps || 0) % (status.diloco.inner_steps_config || 500))
    : 0;

  return (
    <>
      <SEO title="Training Status" description="Monitor the decentralized training progress of the NeuroLLM model." />
      <div className="min-h-screen bg-slate-950 pt-28 pb-12 px-6">
      <div className="container mx-auto max-w-7xl">
        {/* Header */}
        <div className="mb-8 flex flex-col md:flex-row md:items-end md:justify-between gap-4">
          <div>
            <h1 className="text-4xl font-bold text-white mb-2">Training Status</h1>
            <p className="text-slate-400">Real-time decentralized training progress</p>
          </div>
          <div className="flex items-center gap-3">
            {/* Live indicator */}
            <div className="flex items-center gap-2 text-sm text-slate-400">
              {status.is_training && (
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                </span>
              )}
              <span>{status.is_training ? 'Live' : 'Paused'}</span>
              <span className="text-slate-600">•</span>
              <span>Updated {lastUpdate?.toLocaleTimeString()}</span>
            </div>
            <button
              onClick={handleRefresh}
              disabled={refreshing}
              className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {/* Status badges */}
        <div className="flex flex-wrap gap-2 mb-6">
          <StatusBadge 
            type={getStatusType(status.training_verified)}
            label={status.training_verified ? 'Verified' : 'Verifying...'}
          />
          <StatusBadge 
            type={status.is_converging ? 'success' : 'warning'}
            label={status.is_converging ? 'Converging' : 'Syncing...'}
          />
          <StatusBadge 
            type={status.loss_trend === 'improving' ? 'success' : status.loss_trend === 'stable' ? 'warning' : 'error'}
            label={`Loss ${status.loss_trend}`}
          />
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <StatCard
            icon={Users}
            label="Active Nodes"
            value={status.total_nodes_training}
            subtitle="Contributing compute"
          />
          <StatCard
            icon={Activity}
            label="Global Loss"
            value={formatLoss(status.global_loss)}
            subtitle="Network average"
            trend={getTrendFromLossTrend(status.loss_trend)}
          />
          <StatCard
            icon={Zap}
            label="Total Steps"
            value={status.total_training_steps.toLocaleString()}
            subtitle="Training iterations"
          />
          <StatCard
            icon={Database}
            label="Data Shards"
            value={status.data_shards_covered}
            subtitle="Unique datasets"
          />
        </div>

        {/* Two column layout for convergence and DiLoCo */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Convergence Card */}
          <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
            <div className="flex items-center gap-2 mb-4">
              <Layers className="w-5 h-5 text-cyan-400" />
              <h3 className="text-lg font-semibold text-white">Model Convergence</h3>
            </div>
            
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm text-slate-400">Hash Agreement</span>
              <span className={`text-sm font-medium ${
                hashAgreementPercent >= 80 ? 'text-emerald-400' : 
                hashAgreementPercent >= 50 ? 'text-amber-400' : 'text-red-400'
              }`}>
                {hashAgreementPercent}%
              </span>
            </div>
            
            <div className="h-2 bg-slate-800 rounded-full overflow-hidden mb-3">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${hashAgreementPercent}%` }}
                transition={{ duration: 0.5 }}
                className={`h-full rounded-full ${
                  hashAgreementPercent >= 80 ? 'bg-emerald-500' : 
                  hashAgreementPercent >= 50 ? 'bg-amber-500' : 'bg-red-500'
                }`}
              />
            </div>
            
            <p className="text-xs text-slate-500">
              {hashAgreementPercent >= 80 
                ? 'All nodes training the same model'
                : hashAgreementPercent >= 50 
                ? 'Nodes are synchronizing'
                : 'Nodes have diverged - check sync'
              }
            </p>
          </div>

          {/* DiLoCo Card */}
          {status.diloco?.enabled && (
            <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
              <div className="flex items-center gap-2 mb-4">
                <Clock className="w-5 h-5 text-cyan-400" />
                <h3 className="text-lg font-semibold text-white">DiLoCo Protocol</h3>
                <span className="px-2 py-0.5 bg-emerald-500/10 text-emerald-400 text-xs font-medium rounded-full">
                  Active
                </span>
              </div>
              
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <p className="text-2xl font-semibold text-white">{status.diloco?.inner_steps_config || 500}</p>
                  <p className="text-xs text-slate-400">Inner Steps</p>
                </div>
                <div>
                  <p className="text-2xl font-semibold text-white">{status.diloco?.outer_steps_completed || 0}</p>
                  <p className="text-xs text-slate-400">Outer Syncs</p>
                </div>
                <div>
                  <p className="text-2xl font-semibold text-white">{stepsUntilSync}</p>
                  <p className="text-xs text-slate-400">Until Sync</p>
                </div>
              </div>
              
              <div className="mt-4 pt-4 border-t border-slate-800">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-400">Sync Success Rate</span>
                  <span className="text-white font-medium">
                    {Math.round((status.sync_success_rate || 0) * 100)}%
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Active Nodes List */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden">
          <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between">
            <h3 className="text-lg font-semibold text-white">Active Training Nodes</h3>
            <span className="text-sm text-slate-400">{status.nodes?.length || 0} online</span>
          </div>
          
          {(status.nodes?.length || 0) > 0 ? (
            <div className="divide-y divide-slate-800">
              {(status.nodes || []).map((node, idx) => (
                <div key={node.node_id || idx} className="px-6 py-4 flex items-center justify-between hover:bg-slate-800/30 transition-colors">
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500/20 to-purple-500/20 border border-slate-700 flex items-center justify-center">
                      <Server className="w-5 h-5 text-cyan-400" />
                    </div>
                    <div>
                      <p className="font-mono text-sm text-white">{node.node_id || 'unknown'}...</p>
                      <p className="text-xs text-slate-500">
                        {(node.training_rounds || 0).toLocaleString()} steps completed
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-slate-400">Current Loss</p>
                    <p className="font-mono text-lg text-cyan-400">
                      {formatLoss(node.current_loss)}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="px-6 py-12 text-center">
              <Server className="w-10 h-10 text-slate-600 mx-auto mb-3" />
              <p className="text-slate-400">No active training nodes</p>
              <p className="text-sm text-slate-500 mt-1">Nodes will appear here when they start training</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="mt-6 text-center text-sm text-slate-500">
          Auto-refreshes every 10 seconds • Data from {status.source || 'local node'}
        </div>
      </div>
    </div>
    </>
  );
};

export default GlobalLLMStatus;
