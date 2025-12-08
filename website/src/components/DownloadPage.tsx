import { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Terminal,
  Copy,
  Check,
  Cpu,
  Zap,
  Monitor,
  Globe,
  Shield,
  ChevronRight,
  ExternalLink
} from 'lucide-react';
import { SEO } from './SEO';

type Platform = 'any' | 'nvidia' | 'apple';

export const DownloadPage = () => {
  const [selectedPlatform, setSelectedPlatform] = useState<Platform>('any');
  const [copiedCommand, setCopiedCommand] = useState<string | null>(null);

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCommand(id);
    setTimeout(() => setCopiedCommand(null), 2000);
  };

  const platforms = [
    {
      id: 'any' as Platform,
      name: 'CPU / Any',
      icon: <Cpu className="w-6 h-6" />,
      description: 'Works everywhere',
      gradient: 'from-blue-500 to-cyan-600',
    },
    {
      id: 'nvidia' as Platform,
      name: 'NVIDIA GPU',
      icon: <Zap className="w-6 h-6" />,
      description: '10x faster training',
      gradient: 'from-green-500 to-emerald-600',
    },
    {
      id: 'apple' as Platform,
      name: 'Apple Silicon',
      icon: <Monitor className="w-6 h-6" />,
      description: 'M1/M2/M3/M4 GPU',
      gradient: 'from-purple-500 to-pink-600',
    },
  ];

  const getInstallCommands = () => {
    switch (selectedPlatform) {
      case 'nvidia':
        return [
          { id: 'torch-cuda', label: 'Install PyTorch with CUDA', cmd: 'pip install torch --index-url https://download.pytorch.org/whl/cu118' },
          { id: 'neuroshard', label: 'Install NeuroShard', cmd: 'pip install nexaroa' },
        ];
      case 'apple':
        return [
          { id: 'neuroshard', label: 'Install NeuroShard (MPS auto-detected)', cmd: 'pip install nexaroa' },
        ];
      default:
        return [
          { id: 'neuroshard', label: 'Install NeuroShard', cmd: 'pip install nexaroa' },
        ];
    }
  };

  const runCommand = 'neuroshard --token YOUR_TOKEN';

  return (
    <>
      <SEO title="Download" description="Download and install the NeuroShard node to start contributing to the network." />
      <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950">
      {/* Background Effects */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/5 rounded-full blur-3xl"></div>
      </div>

      <div className="relative z-10 container mx-auto px-4 sm:px-6 lg:px-8 pt-24 pb-20">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-slate-800/50 border border-slate-700 backdrop-blur-sm mb-6">
            <div className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
            </div>
            <span className="text-sm font-medium text-cyan-400">
              Install via pip
            </span>
          </div>

          <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold text-white mb-6 tracking-tight">
            Get Started with
            <span className="block text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600">
              NeuroShard
            </span>
          </h1>

          <p className="text-xl text-slate-400 max-w-2xl mx-auto">
            One command to join the decentralized AI network. Start earning NEURO tokens by sharing your computing power.
          </p>
        </motion.div>

        {/* Platform Selection */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="max-w-4xl mx-auto mb-8"
        >
          <h2 className="text-lg font-semibold text-white mb-4 text-center">Select your platform</h2>
          <div className="grid grid-cols-3 gap-4">
            {platforms.map((platform) => (
              <button
                key={platform.id}
                onClick={() => setSelectedPlatform(platform.id)}
                className={`p-4 rounded-xl border transition-all ${
                  selectedPlatform === platform.id
                    ? `bg-gradient-to-br ${platform.gradient} border-transparent text-white shadow-lg`
                    : 'bg-slate-900/50 border-slate-800 text-slate-400 hover:border-slate-700'
                }`}
              >
                <div className="flex flex-col items-center gap-2">
                  {platform.icon}
                  <span className="font-semibold text-sm">{platform.name}</span>
                  <span className="text-xs opacity-80">{platform.description}</span>
                </div>
              </button>
            ))}
          </div>
        </motion.div>

        {/* Installation Commands */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="max-w-4xl mx-auto mb-12"
        >
          <div className="bg-slate-900 rounded-2xl border border-slate-800 overflow-hidden">
            {/* Terminal Header */}
            <div className="flex items-center gap-2 px-4 py-3 bg-slate-950 border-b border-slate-800">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-red-500"></div>
                <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
              </div>
              <span className="text-xs text-slate-500 ml-2">Terminal</span>
            </div>

            {/* Commands */}
            <div className="p-6 space-y-4">
              {/* Step 1: Install */}
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span className="w-6 h-6 rounded-full bg-cyan-500/20 text-cyan-400 flex items-center justify-center text-xs font-bold">1</span>
                  <span className="text-sm text-slate-400">Install</span>
                </div>
                {getInstallCommands().map((command, idx) => (
                  <div key={command.id} className={`${idx > 0 ? 'mt-2' : ''}`}>
                    {getInstallCommands().length > 1 && (
                      <p className="text-xs text-slate-500 mb-1 ml-8">{command.label}</p>
                    )}
                    <div className="flex items-center gap-2 bg-slate-950 rounded-lg p-3 border border-slate-800 overflow-x-auto">
                      <span className="text-green-400 font-mono">$</span>
                      <code className="flex-1 text-cyan-400 font-mono text-sm">{command.cmd}</code>
                      <button
                        onClick={() => copyToClipboard(command.cmd, command.id)}
                        className="p-1.5 rounded hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
                      >
                        {copiedCommand === command.id ? (
                          <Check className="w-4 h-4 text-green-400" />
                        ) : (
                          <Copy className="w-4 h-4" />
                        )}
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              {/* Step 2: Run */}
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span className="w-6 h-6 rounded-full bg-purple-500/20 text-purple-400 flex items-center justify-center text-xs font-bold">2</span>
                  <span className="text-sm text-slate-400">Run your node</span>
                </div>
                <div className="flex items-center gap-2 bg-slate-950 rounded-lg p-3 border border-slate-800">
                  <span className="text-green-400 font-mono">$</span>
                  <code className="flex-1 text-cyan-400 font-mono text-sm">{runCommand}</code>
                  <button
                    onClick={() => copyToClipboard(runCommand, 'run')}
                    className="p-1.5 rounded hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
                  >
                    {copiedCommand === 'run' ? (
                      <Check className="w-4 h-4 text-green-400" />
                    ) : (
                      <Copy className="w-4 h-4" />
                    )}
                  </button>
                </div>
                <p className="text-xs text-slate-500 mt-2 ml-8">
                  Get your token at <a href="https://neuroshard.com/register" className="text-cyan-400 hover:underline">neuroshard.com/register</a>
                </p>
              </div>

              {/* Step 3: Dashboard */}
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span className="w-6 h-6 rounded-full bg-green-500/20 text-green-400 flex items-center justify-center text-xs font-bold">3</span>
                  <span className="text-sm text-slate-400">View your dashboard</span>
                </div>
                <div className="flex items-center gap-2 bg-slate-950 rounded-lg p-3 border border-slate-800">
                  <Globe className="w-4 h-4 text-slate-500" />
                  <code className="flex-1 text-slate-300 font-mono text-sm">http://localhost:8000</code>
                  <span className="text-xs text-slate-500">Opens automatically</span>
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* CLI Options */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="max-w-4xl mx-auto mb-16"
        >
          <h2 className="text-2xl font-bold text-white mb-6 text-center">Command Options</h2>

          <div className="bg-slate-900/50 border border-slate-800 rounded-xl overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-800">
                  <th className="text-left text-slate-400 font-medium px-6 py-3">Option</th>
                  <th className="text-left text-slate-400 font-medium px-6 py-3">Description</th>
                  <th className="text-left text-slate-400 font-medium px-6 py-3">Default</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                <tr>
                  <td className="px-6 py-3 font-mono text-cyan-400">--token</td>
                  <td className="px-6 py-3 text-slate-300">Your wallet recovery phrase</td>
                  <td className="px-6 py-3 text-slate-500">Required</td>
                </tr>
                <tr>
                  <td className="px-6 py-3 font-mono text-cyan-400">--port</td>
                  <td className="px-6 py-3 text-slate-300">HTTP/Dashboard port</td>
                  <td className="px-6 py-3 text-slate-500">8000</td>
                </tr>
                <tr>
                  <td className="px-6 py-3 font-mono text-cyan-400">--memory</td>
                  <td className="px-6 py-3 text-slate-300">Max memory limit (MB)</td>
                  <td className="px-6 py-3 text-slate-500">4096</td>
                </tr>
                <tr>
                  <td className="px-6 py-3 font-mono text-cyan-400">--cpu-threads</td>
                  <td className="px-6 py-3 text-slate-300">Max CPU threads</td>
                  <td className="px-6 py-3 text-slate-500">4</td>
                </tr>
                <tr>
                  <td className="px-6 py-3 font-mono text-cyan-400">--no-training</td>
                  <td className="px-6 py-3 text-slate-300">Inference only mode</td>
                  <td className="px-6 py-3 text-slate-500">false</td>
                </tr>
                <tr>
                  <td className="px-6 py-3 font-mono text-cyan-400">--headless</td>
                  <td className="px-6 py-3 text-slate-300">Don't auto-open browser</td>
                  <td className="px-6 py-3 text-slate-500">false</td>
                </tr>
                <tr>
                  <td className="px-6 py-3 font-mono text-cyan-400">--daemon</td>
                  <td className="px-6 py-3 text-slate-300">Run as background service</td>
                  <td className="px-6 py-3 text-slate-500">false</td>
                </tr>
                <tr>
                  <td className="px-6 py-3 font-mono text-cyan-400">--stop</td>
                  <td className="px-6 py-3 text-slate-300">Stop the background daemon</td>
                  <td className="px-6 py-3 text-slate-500">-</td>
                </tr>
                <tr>
                  <td className="px-6 py-3 font-mono text-cyan-400">--status</td>
                  <td className="px-6 py-3 text-slate-300">Check if daemon is running</td>
                  <td className="px-6 py-3 text-slate-500">-</td>
                </tr>
                <tr>
                  <td className="px-6 py-3 font-mono text-cyan-400">--logs</td>
                  <td className="px-6 py-3 text-slate-300">View daemon logs</td>
                  <td className="px-6 py-3 text-slate-500">-</td>
                </tr>
              </tbody>
            </table>
          </div>

          <p className="text-center text-sm text-slate-500 mt-4">
            Full documentation at{' '}
            <a href="https://docs.neuroshard.com" className="text-cyan-400 hover:underline inline-flex items-center gap-1">
              docs.neuroshard.com <ExternalLink className="w-3 h-3" />
            </a>
          </p>
        </motion.div>

        {/* Features Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="max-w-4xl mx-auto"
        >
          <h2 className="text-2xl font-bold text-white mb-8 text-center">What You Get</h2>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center mb-4">
                <Terminal className="w-6 h-6 text-white" />
              </div>
              <h3 className="font-semibold text-white mb-2">Web Dashboard</h3>
              <p className="text-sm text-slate-400">
                Monitor your node status, training progress, NEURO balance, and resource usage from a beautiful local dashboard.
              </p>
            </div>

            <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center mb-4">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <h3 className="font-semibold text-white mb-2">Automatic GPU Detection</h3>
              <p className="text-sm text-slate-400">
                NVIDIA CUDA and Apple Metal are auto-detected. Training is 10x faster with GPU acceleration.
              </p>
            </div>

            <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center mb-4">
                <Globe className="w-6 h-6 text-white" />
              </div>
              <h3 className="font-semibold text-white mb-2">Swarm Architecture</h3>
              <p className="text-sm text-slate-400">
                Fault-tolerant multipath routing ensures your work is never lost. DiLoCo reduces sync bandwidth by 90%.
              </p>
            </div>

            <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-orange-500 to-red-600 flex items-center justify-center mb-4">
                <Shield className="w-6 h-6 text-white" />
              </div>
              <h3 className="font-semibold text-white mb-2">ECDSA Cryptography</h3>
              <p className="text-sm text-slate-400">
                All Proof of Neural Work is cryptographically signed. Your rewards are verifiable and tamper-proof.
              </p>
            </div>
          </div>
        </motion.div>

        {/* System Requirements */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="max-w-4xl mx-auto mt-16"
        >
          <h3 className="text-xl font-bold text-white mb-6 text-center">System Requirements</h3>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
              <h4 className="font-semibold text-white mb-4 flex items-center gap-2">
                <Cpu className="w-5 h-5 text-cyan-400" />
                Minimum
              </h4>
              <ul className="space-y-2 text-sm text-slate-400">
                <li className="flex items-center gap-2">
                  <ChevronRight className="w-4 h-4 text-cyan-400" />
                  Python 3.9+
                </li>
                <li className="flex items-center gap-2">
                  <ChevronRight className="w-4 h-4 text-cyan-400" />
                  4GB RAM
                </li>
                <li className="flex items-center gap-2">
                  <ChevronRight className="w-4 h-4 text-cyan-400" />
                  Dual-core CPU
                </li>
                <li className="flex items-center gap-2">
                  <ChevronRight className="w-4 h-4 text-cyan-400" />
                  Stable internet connection
                </li>
              </ul>
            </div>

            <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
              <h4 className="font-semibold text-white mb-4 flex items-center gap-2">
                <Zap className="w-5 h-5 text-green-400" />
                Recommended
              </h4>
              <ul className="space-y-2 text-sm text-slate-400">
                <li className="flex items-center gap-2">
                  <ChevronRight className="w-4 h-4 text-green-400" />
                  8GB+ RAM
                </li>
                <li className="flex items-center gap-2">
                  <ChevronRight className="w-4 h-4 text-green-400" />
                  NVIDIA GPU (GTX 1060+) or Apple Silicon
                </li>
                <li className="flex items-center gap-2">
                  <ChevronRight className="w-4 h-4 text-green-400" />
                  SSD storage
                </li>
                <li className="flex items-center gap-2">
                  <ChevronRight className="w-4 h-4 text-green-400" />
                  Always-on machine
                </li>
              </ul>
            </div>
          </div>
        </motion.div>

        {/* PyPI Badge */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="text-center mt-16"
        >
          <a
            href="https://pypi.org/project/nexaroa/"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-3 px-6 py-3 rounded-xl bg-slate-900/50 border border-slate-800 hover:border-slate-700 transition-colors"
          >
            <img src="https://badge.fury.io/py/nexaroa.svg" alt="PyPI version" className="h-5" />
            <span className="text-slate-400 text-sm">View on PyPI</span>
            <ExternalLink className="w-4 h-4 text-slate-500" />
          </a>
        </motion.div>
      </div>
    </div>
    </>
  );
};
