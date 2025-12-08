import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Cpu, Zap, Wifi, Monitor, ArrowRight,
  Check, Copy, Link2,
  AlertCircle, Share2, RefreshCw
} from 'lucide-react';
import axios from 'axios';
import { API_URL } from '../config/api';
import { SEO } from './SEO';

type Step = 'hardware' | 'calculating' | 'results';

interface HardwareData {
  email: string;
  gpu_model: string;
  gpu_vram: number | null;
  ram_gb: number;
  internet_speed: number | null;
  operating_system: string;
  referral_code: string;
}

interface WaitlistResult {
  id: number;
  email: string;
  referral_code: string;
  referral_url: string;
  position: number;
  estimated_daily_neuro: number;
  hardware_tier: string;
  hardware_score: number;
  priority_score: number;
  status: string;
  message: string;
}

const GPU_OPTIONS = [
  { value: '', label: 'Select your GPU...' },
  { value: 'none', label: 'No GPU / CPU Only' },
  // NVIDIA
  { value: 'RTX 4090', label: 'NVIDIA RTX 4090' },
  { value: 'RTX 4080', label: 'NVIDIA RTX 4080' },
  { value: 'RTX 4070 Ti', label: 'NVIDIA RTX 4070 Ti' },
  { value: 'RTX 4070', label: 'NVIDIA RTX 4070' },
  { value: 'RTX 4060 Ti', label: 'NVIDIA RTX 4060 Ti' },
  { value: 'RTX 4060', label: 'NVIDIA RTX 4060' },
  { value: 'RTX 3090 Ti', label: 'NVIDIA RTX 3090 Ti' },
  { value: 'RTX 3090', label: 'NVIDIA RTX 3090' },
  { value: 'RTX 3080 Ti', label: 'NVIDIA RTX 3080 Ti' },
  { value: 'RTX 3080', label: 'NVIDIA RTX 3080' },
  { value: 'RTX 3070 Ti', label: 'NVIDIA RTX 3070 Ti' },
  { value: 'RTX 3070', label: 'NVIDIA RTX 3070' },
  { value: 'RTX 3060 Ti', label: 'NVIDIA RTX 3060 Ti' },
  { value: 'RTX 3060', label: 'NVIDIA RTX 3060' },
  { value: 'RTX 2080 Ti', label: 'NVIDIA RTX 2080 Ti' },
  { value: 'RTX 2080', label: 'NVIDIA RTX 2080' },
  { value: 'RTX 2070', label: 'NVIDIA RTX 2070' },
  { value: 'RTX 2060', label: 'NVIDIA RTX 2060' },
  { value: 'GTX 1080 Ti', label: 'NVIDIA GTX 1080 Ti' },
  { value: 'GTX 1080', label: 'NVIDIA GTX 1080' },
  { value: 'GTX 1070', label: 'NVIDIA GTX 1070' },
  { value: 'GTX 1060', label: 'NVIDIA GTX 1060' },
  // AMD
  { value: 'RX 7900 XTX', label: 'AMD RX 7900 XTX' },
  { value: 'RX 7900 XT', label: 'AMD RX 7900 XT' },
  { value: 'RX 7800 XT', label: 'AMD RX 7800 XT' },
  { value: 'RX 7700 XT', label: 'AMD RX 7700 XT' },
  { value: 'RX 7600', label: 'AMD RX 7600' },
  { value: 'RX 6900 XT', label: 'AMD RX 6900 XT' },
  { value: 'RX 6800 XT', label: 'AMD RX 6800 XT' },
  { value: 'RX 6700 XT', label: 'AMD RX 6700 XT' },
  // Apple Silicon
  { value: 'M3 Ultra', label: 'Apple M3 Ultra' },
  { value: 'M3 Max', label: 'Apple M3 Max' },
  { value: 'M3 Pro', label: 'Apple M3 Pro' },
  { value: 'M3', label: 'Apple M3' },
  { value: 'M2 Ultra', label: 'Apple M2 Ultra' },
  { value: 'M2 Max', label: 'Apple M2 Max' },
  { value: 'M2 Pro', label: 'Apple M2 Pro' },
  { value: 'M2', label: 'Apple M2' },
  { value: 'M1 Ultra', label: 'Apple M1 Ultra' },
  { value: 'M1 Max', label: 'Apple M1 Max' },
  { value: 'M1 Pro', label: 'Apple M1 Pro' },
  { value: 'M1', label: 'Apple M1' },
  // Pro/Enterprise
  { value: 'A100', label: 'NVIDIA A100' },
  { value: 'H100', label: 'NVIDIA H100' },
  { value: 'A6000', label: 'NVIDIA A6000' },
  { value: 'T4', label: 'NVIDIA T4' },
  { value: 'other', label: 'Other GPU' },
];

const RAM_OPTIONS = [
  { value: 8, label: '8 GB' },
  { value: 16, label: '16 GB' },
  { value: 32, label: '32 GB' },
  { value: 64, label: '64 GB' },
  { value: 128, label: '128 GB' },
  { value: 256, label: '256 GB+' },
];

const INTERNET_OPTIONS = [
  { value: 25, label: '25 Mbps (Basic)' },
  { value: 50, label: '50 Mbps' },
  { value: 100, label: '100 Mbps' },
  { value: 200, label: '200 Mbps' },
  { value: 500, label: '500 Mbps' },
  { value: 1000, label: '1 Gbps+' },
];

const OS_OPTIONS = [
  { value: 'Windows', label: 'Windows' },
  { value: 'macOS', label: 'macOS' },
  { value: 'Linux', label: 'Linux' },
];

export const WaitlistSignup = () => {
  const [searchParams] = useSearchParams();
  
  const [step, setStep] = useState<Step>('hardware');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);
  
  const [referralValid, setReferralValid] = useState<boolean | null>(null);
  const [referralInfo, setReferralInfo] = useState<string>('');
  
  const [hardware, setHardware] = useState<HardwareData>({
    email: '',
    gpu_model: '',
    gpu_vram: null,
    ram_gb: 16,
    internet_speed: 100,
    operating_system: 'Windows',
    referral_code: searchParams.get('ref') || '',
  });
  
  const [result, setResult] = useState<WaitlistResult | null>(null);
  
  // Check referral code validity
  useEffect(() => {
    const checkReferral = async () => {
      if (hardware.referral_code && hardware.referral_code.length >= 6) {
        try {
          const response = await axios.get(
            `${API_URL}/api/waitlist/check-referral?code=${hardware.referral_code}`
          );
          setReferralValid(response.data.valid);
          setReferralInfo(response.data.message);
        } catch {
          setReferralValid(false);
          setReferralInfo('Invalid referral code');
        }
      } else {
        setReferralValid(null);
        setReferralInfo('');
      }
    };
    
    const timer = setTimeout(checkReferral, 500);
    return () => clearTimeout(timer);
  }, [hardware.referral_code]);
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setStep('calculating');
    setLoading(true);
    
    // Simulate calculation animation
    await new Promise(resolve => setTimeout(resolve, 2500));
    
    try {
      const response = await axios.post(`${API_URL}/api/waitlist/signup`, {
        email: hardware.email,
        gpu_model: hardware.gpu_model || null,
        gpu_vram: hardware.gpu_vram,
        ram_gb: hardware.ram_gb,
        internet_speed: hardware.internet_speed,
        operating_system: hardware.operating_system,
        referral_code: hardware.referral_code || null,
      });
      
      setResult(response.data);
      setStep('results');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to join waitlist. Please try again.');
      setStep('hardware');
    } finally {
      setLoading(false);
    }
  };
  
  const copyReferralLink = () => {
    if (result?.referral_url) {
      navigator.clipboard.writeText(result.referral_url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };
  
  const shareOnTwitter = () => {
    if (result) {
      const text = `I just reserved my spot to contribute to @NeuroShardAI - the decentralized AI network. Join the distributed intelligence revolution:`;
      const url = result.referral_url;
      window.open(`https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(url)}`, '_blank');
    }
  };
  
  // Hardware Form Step
  if (step === 'hardware') {
    return (
      <>
        <SEO title="Join the Waitlist" description="Register your hardware and join the NeuroShard waitlist to earn NEURO tokens." />
        <div className="min-h-screen bg-slate-950 pt-28 pb-12 px-6">
        <div className="container mx-auto max-w-2xl">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-4xl font-bold text-white mb-2">Join the Network</h1>
            <p className="text-slate-400">Register your hardware to reserve your spot in the NeuroShard network.</p>
          </div>
          
          {/* Form Card */}
          <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
            {error && (
              <motion.div 
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-start gap-3"
              >
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                <p className="text-red-300 text-sm">{error}</p>
              </motion.div>
            )}
            
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Email */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Email Address
                </label>
                <input
                  type="email"
                  value={hardware.email}
                  onChange={(e) => setHardware({ ...hardware, email: e.target.value })}
                  className="w-full px-4 py-3 bg-slate-950 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all"
                  placeholder="you@example.com"
                  required
                />
              </div>
              
              {/* GPU */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="flex items-center gap-2 text-sm font-medium text-slate-300 mb-2">
                    <Cpu className="w-4 h-4 text-cyan-400" />
                    GPU Model
                  </label>
                  <select
                    value={hardware.gpu_model}
                    onChange={(e) => setHardware({ ...hardware, gpu_model: e.target.value })}
                    className="w-full px-4 py-3 bg-slate-950 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all appearance-none cursor-pointer"
                  >
                    {GPU_OPTIONS.map(opt => (
                      <option key={opt.value} value={opt.value}>{opt.label}</option>
                    ))}
                  </select>
                </div>
                
                <div>
                  <label className="flex items-center gap-2 text-sm font-medium text-slate-300 mb-2">
                    <Monitor className="w-4 h-4 text-slate-400" />
                    VRAM (if applicable)
                  </label>
                  <select
                    value={hardware.gpu_vram || ''}
                    onChange={(e) => setHardware({ ...hardware, gpu_vram: e.target.value ? parseInt(e.target.value) : null })}
                    className="w-full px-4 py-3 bg-slate-950 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all appearance-none cursor-pointer"
                    disabled={!hardware.gpu_model || hardware.gpu_model === 'none'}
                  >
                    <option value="">Select VRAM...</option>
                    <option value="4">4 GB</option>
                    <option value="6">6 GB</option>
                    <option value="8">8 GB</option>
                    <option value="10">10 GB</option>
                    <option value="12">12 GB</option>
                    <option value="16">16 GB</option>
                    <option value="24">24 GB</option>
                    <option value="32">32 GB</option>
                    <option value="48">48 GB</option>
                    <option value="80">80 GB+</option>
                  </select>
                </div>
              </div>
              
              {/* RAM & Internet */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="flex items-center gap-2 text-sm font-medium text-slate-300 mb-2">
                    <Zap className="w-4 h-4 text-emerald-400" />
                    System RAM
                  </label>
                  <select
                    value={hardware.ram_gb}
                    onChange={(e) => setHardware({ ...hardware, ram_gb: parseInt(e.target.value) })}
                    className="w-full px-4 py-3 bg-slate-950 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all appearance-none cursor-pointer"
                    required
                  >
                    {RAM_OPTIONS.map(opt => (
                      <option key={opt.value} value={opt.value}>{opt.label}</option>
                    ))}
                  </select>
                </div>
                
                <div>
                  <label className="flex items-center gap-2 text-sm font-medium text-slate-300 mb-2">
                    <Wifi className="w-4 h-4 text-blue-400" />
                    Internet Speed
                  </label>
                  <select
                    value={hardware.internet_speed || ''}
                    onChange={(e) => setHardware({ ...hardware, internet_speed: e.target.value ? parseInt(e.target.value) : null })}
                    className="w-full px-4 py-3 bg-slate-950 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all appearance-none cursor-pointer"
                  >
                    {INTERNET_OPTIONS.map(opt => (
                      <option key={opt.value} value={opt.value}>{opt.label}</option>
                    ))}
                  </select>
                </div>
              </div>
              
              {/* OS */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Operating System
                </label>
                <div className="flex gap-3">
                  {OS_OPTIONS.map(opt => (
                    <button
                      key={opt.value}
                      type="button"
                      onClick={() => setHardware({ ...hardware, operating_system: opt.value })}
                      className={`flex-1 py-3 px-4 rounded-lg border font-medium transition-all ${
                        hardware.operating_system === opt.value
                          ? 'bg-cyan-500/10 border-cyan-500/30 text-cyan-400'
                          : 'bg-slate-950 border-slate-700 text-slate-400 hover:border-slate-600'
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>
              
              {/* Referral Code */}
              <div>
                <label className="flex items-center gap-2 text-sm font-medium text-slate-300 mb-2">
                  <Link2 className="w-4 h-4 text-slate-400" />
                  Referral Code (Optional)
                </label>
                <div className="relative">
                  <input
                    type="text"
                    value={hardware.referral_code}
                    onChange={(e) => setHardware({ ...hardware, referral_code: e.target.value.toUpperCase() })}
                    className={`w-full px-4 py-3 bg-slate-950 border rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all ${
                      referralValid === true ? 'border-emerald-500/50' :
                      referralValid === false ? 'border-red-500/50' :
                      'border-slate-700'
                    }`}
                    placeholder="Enter friend's code for priority boost"
                    maxLength={12}
                  />
                  {referralValid !== null && (
                    <div className={`absolute right-3 top-1/2 -translate-y-1/2 text-sm ${
                      referralValid ? 'text-emerald-400' : 'text-red-400'
                    }`}>
                      {referralValid ? <Check className="w-5 h-5" /> : <AlertCircle className="w-5 h-5" />}
                    </div>
                  )}
                </div>
                {referralInfo && (
                  <p className={`mt-1.5 text-xs ${referralValid ? 'text-emerald-400' : 'text-red-400'}`}>
                    {referralInfo}
                  </p>
                )}
              </div>
              
              {/* Submit */}
              <button
                type="submit"
                disabled={loading || !hardware.email}
                className="w-full py-3 bg-cyan-500 hover:bg-cyan-400 text-white font-semibold rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                Reserve My Spot
                <ArrowRight className="w-4 h-4" />
              </button>
            </form>
            
            <p className="text-center text-slate-500 text-sm mt-6">
              Already on the waitlist? <a href="/login" className="text-cyan-400 hover:underline">Check your status</a>
            </p>
          </div>
        </div>
      </div>
      </>
    );
  }
  
  // Calculating Animation Step
  if (step === 'calculating') {
    return (
      <div className="min-h-screen bg-slate-950 pt-28 pb-12 px-6 flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="w-8 h-8 text-cyan-400 animate-spin mx-auto mb-4" />
          <p className="text-white font-medium mb-2">Registering your hardware...</p>
          <p className="text-slate-500 text-sm">This will only take a moment.</p>
        </div>
      </div>
    );
  }
  
  // Results Step
  if (step === 'results' && result) {
    return (
      <>
        <SEO title="Waitlist Confirmed" description="You are on the list! Share your referral link to boost your priority." />
        <div className="min-h-screen bg-slate-950 pt-28 pb-12 px-6">
        <div className="container mx-auto max-w-2xl">
          {/* Success Header */}
          <div className="mb-8 flex flex-col items-center">
            <h1 className="text-4xl font-bold text-white mb-2">You're on the List</h1>
          </div>
          
          {/* Estimated Earnings Card */}
          <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6 mb-6">
            <div className="text-center">
              <p className="text-sm text-slate-400 mb-3">Estimated Daily Earnings</p>
              <div className="flex items-baseline justify-center gap-1">
                <span className="text-4xl font-bold text-white">
                  {result.estimated_daily_neuro.toFixed(2)}
                </span>
                <span className="text-xl text-slate-400">NEURO</span>
              </div>
              <p className="text-xs text-slate-500 mt-2">
                Based on your hardware configuration
              </p>
            </div>
          </div>
          
          {/* Neuro Link Card */}
          <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6 mb-6">
            <div className="flex items-center gap-2 mb-4">
              <Link2 className="w-5 h-5 text-cyan-400" />
              <h3 className="text-lg font-semibold text-white">Your Neuro Link</h3>
            </div>
            
            <p className="text-sm text-slate-400 mb-4">
              Share this link to move up in the queue. Each referral gives you +10 priority points.
            </p>
            
            <div className="flex gap-2 mb-4">
              <div className="flex-1 bg-slate-950 border border-slate-700 rounded-lg px-4 py-3 font-mono text-cyan-400 text-sm truncate">
                {result.referral_url}
              </div>
              <button
                onClick={copyReferralLink}
                className={`px-4 py-3 rounded-lg font-medium transition-all flex items-center gap-2 ${
                  copied 
                    ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700'
                }`}
              >
                {copied ? <Check className="w-5 h-5" /> : <Copy className="w-5 h-5" />}
              </button>
            </div>
            
            <button
              onClick={shareOnTwitter}
              className="w-full py-3 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
            >
              <Share2 className="w-4 h-4" />
              Share on X
            </button>
          </div>
          
          {/* What's Next Card */}
          <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-white mb-4">What happens next?</h3>
            <div className="space-y-4 text-sm">
              <div className="flex items-start gap-3">
                <span className="w-6 h-6 rounded-full bg-slate-800 text-slate-400 flex items-center justify-center text-xs font-medium flex-shrink-0">1</span>
                <p className="text-slate-400">We'll review your hardware and approve your node for the network.</p>
              </div>
              <div className="flex items-start gap-3">
                <span className="w-6 h-6 rounded-full bg-slate-800 text-slate-400 flex items-center justify-center text-xs font-medium flex-shrink-0">2</span>
                <p className="text-slate-400">You'll receive an email when approved with instructions to complete registration.</p>
              </div>
              <div className="flex items-start gap-3">
                <span className="w-6 h-6 rounded-full bg-slate-800 text-slate-400 flex items-center justify-center text-xs font-medium flex-shrink-0">3</span>
                <p className="text-slate-400">Download the NeuroShard node, create your wallet, and start earning NEURO.</p>
              </div>
            </div>
          </div>
          
          <p className="text-center text-slate-600 text-xs mt-6">
            Earnings estimates are projections based on current network conditions. Actual earnings may vary.
          </p>
        </div>
      </div>
      </>
    );
  }
  
  return null;
};

