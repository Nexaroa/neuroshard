import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Activity, Cpu, Globe, Zap } from 'lucide-react';
import axios from 'axios';
import { API_URL } from '../config/api';

const StatItem = ({ icon: Icon, label, value, color, delay }: any) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay, duration: 0.5 }}
    className="flex flex-col items-center md:items-start p-4 bg-slate-900/40 border border-slate-800/50 rounded-xl backdrop-blur-sm min-w-[140px]"
  >
    <div className={`flex items-center gap-2 mb-2 text-${color}-400`}>
      <Icon className="w-4 h-4" />
      <span className="text-xs font-bold uppercase tracking-wider opacity-80">{label}</span>
    </div>
    <div className="text-2xl font-bold text-white font-mono">
      {value}
    </div>
  </motion.div>
);

export const LiveStats = () => {
  const [stats, setStats] = useState({
    nodes: '0',
    params: '142B',
    tps: '0',
    latency: '--'
  });

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await axios.get(`${API_URL}/api/stats`);
        const data = response.data;
        setStats({
          nodes: data.active_nodes?.toLocaleString() || '0',
          params: data.model_size || '142B',
          tps: data.total_tps?.toLocaleString() || '0',
          latency: data.avg_latency || '--'
        });
      } catch (error) {
        console.error('Failed to fetch live stats:', error);
      }
    };

    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-wrap justify-center gap-4 mt-12 mb-8">
      <StatItem 
        icon={Globe} 
        label="Active Nodes" 
        value={stats.nodes} 
        color="cyan" 
        delay={1.2} 
      />
      <StatItem 
        icon={Cpu} 
        label="Model Size" 
        value={stats.params} 
        color="purple" 
        delay={1.3} 
      />
      <StatItem 
        icon={Zap} 
        label="Network TPS" 
        value={stats.tps} 
        color="yellow" 
        delay={1.4} 
      />
      <StatItem 
        icon={Activity} 
        label="Avg Latency" 
        value={stats.latency} 
        color="green" 
        delay={1.5} 
      />
    </div>
  );
};
