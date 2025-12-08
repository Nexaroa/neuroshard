import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import { motion } from 'framer-motion';
import { API_URL } from '../config/api';
import { AlertTriangle, Copy, Check, Eye, EyeOff, ArrowRight, Terminal, ExternalLink, Sparkles } from 'lucide-react';

type Step = 'account' | 'wallet' | 'import' | 'mnemonic' | 'complete';

interface WalletData {
  mnemonic: string;
  token: string;
  node_id: string;
  wallet_id: string;
}

export const Signup = () => {
  const [step, setStep] = useState<Step>('account');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [wallet, setWallet] = useState<WalletData | null>(null);
  const [copiedMnemonic, setCopiedMnemonic] = useState(false);
  const [copiedToken, setCopiedToken] = useState(false);
  const [showMnemonic, setShowMnemonic] = useState(true);
  const [showToken, setShowToken] = useState(false);
  const [confirmed, setConfirmed] = useState(false);
  const [importSecret, setImportSecret] = useState('');
  const [acceptedTerms, setAcceptedTerms] = useState(false);
  const { login, refreshUser, isAuthenticated, hasWallet, isLoading } = useAuth();
  const navigate = useNavigate();

  // Redirect users who already have a complete account (authenticated + wallet)
  useEffect(() => {
    if (!isLoading && isAuthenticated && hasWallet) {
      navigate('/dashboard', { replace: true });
    }
    // If authenticated but no wallet, skip to wallet step
    if (!isLoading && isAuthenticated && !hasWallet && step === 'account') {
      setStep('wallet');
    }
  }, [isAuthenticated, hasWallet, isLoading, navigate, step]);

  // Step 1: Create Account
  const handleCreateAccount = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    
    try {
      // Create the user account
      await axios.post(`${API_URL}/api/auth/signup`, { email, password });
      
      // Automatically log in
      const formData = new FormData();
      formData.append('username', email);
      formData.append('password', password);
      
      const response = await axios.post(`${API_URL}/api/auth/token`, formData);
      // Pass full response including both access_token and refresh_token
      await login(response.data);
      
      // Move to wallet creation step
      setStep('wallet');
    } catch (err: any) {
      const detail = err.response?.data?.detail || '';
      if (err.response?.status === 400 && detail === 'Email already registered') {
        setError('Email already registered. Please use a different email or log in.');
      } else if (err.response?.status === 403 && detail.includes('waitlist')) {
        // Waitlist-related error - show special message
        setError(detail);
      } else {
        setError(detail || 'Failed to create account');
      }
    } finally {
      setLoading(false);
    }
  };

  // Step 2: Create Wallet
  const handleCreateWallet = async () => {
    setError('');
    setLoading(true);
    
    try {
      const token = localStorage.getItem('token');
      const response = await axios.post(`${API_URL}/api/wallet/create`, {}, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      setWallet(response.data);
      setStep('mnemonic');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to create wallet');
    } finally {
      setLoading(false);
    }
  };

  // Copy mnemonic
  const copyMnemonic = () => {
    if (wallet?.mnemonic) {
      navigator.clipboard.writeText(wallet.mnemonic);
      setCopiedMnemonic(true);
      setTimeout(() => setCopiedMnemonic(false), 2000);
    }
  };

  // Copy token (for runner)
  const copyToken = () => {
    if (wallet?.token) {
      navigator.clipboard.writeText(wallet.token);
      setCopiedToken(true);
      setTimeout(() => setCopiedToken(false), 2000);
    }
  };

  // Import existing wallet
  const handleImportWallet = async () => {
    if (!importSecret.trim()) {
      setError('Please enter your recovery phrase or token');
      return;
    }
    
    setError('');
    setLoading(true);
    
    try {
      const token = localStorage.getItem('token');
      const response = await axios.post(`${API_URL}/api/wallet/connect`, 
        { secret: importSecret.trim() },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      
      // Set wallet data for display (token is the imported secret)
      setWallet({
        mnemonic: importSecret.trim().split(' ').length >= 12 ? importSecret.trim() : '',
        token: response.data.token || importSecret.trim(),
        node_id: response.data.node_id,
        wallet_id: response.data.wallet_id
      });
      
      // Skip mnemonic display for import - go straight to complete
      navigate('/dashboard');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to import wallet');
    } finally {
      setLoading(false);
    }
  };

  // Step 3: Complete registration
  const handleComplete = async () => {
    if (!confirmed) return;
    
    // Refresh user data to get wallet info
    await refreshUser();
    
    navigate('/dashboard');
  };

  // Show loading while checking auth status
  if (isLoading || (isAuthenticated && hasWallet)) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-950">
        <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-cyan-500"></div>
      </div>
    );
  }

  // Step 1: Account Creation
  if (step === 'account') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-950 px-4 pt-20">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-md w-full bg-slate-900 p-8 rounded-2xl border border-slate-800 shadow-xl"
        >
          {/* Progress Indicator */}
          <div className="flex items-center justify-center gap-2 mb-8">
            <div className="w-8 h-8 rounded-full bg-cyan-500 text-white flex items-center justify-center text-sm font-bold">1</div>
            <div className="w-12 h-0.5 bg-slate-700"></div>
            <div className="w-8 h-8 rounded-full bg-slate-700 text-slate-500 flex items-center justify-center text-sm font-bold">2</div>
            <div className="w-12 h-0.5 bg-slate-700"></div>
            <div className="w-8 h-8 rounded-full bg-slate-700 text-slate-500 flex items-center justify-center text-sm font-bold">3</div>
          </div>

          <h2 className="text-3xl font-bold text-white mb-2 text-center">Create Account</h2>
          <p className="text-slate-400 text-center mb-6">Step 1 of 3: Your login credentials</p>
          
          {error && (
            <div className="mb-4 text-center bg-red-500/10 border border-red-500/50 rounded-lg p-4 shadow-[0_0_15px_rgba(239,68,68,0.15)]">
              <p className="text-red-300 text-sm">{error}</p>
              {error.includes('waitlist') && (
                <Link 
                  to="/join" 
                  className="mt-3 inline-flex items-center gap-2 px-4 py-2 bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-400 rounded-lg text-sm font-medium transition-colors"
                >
                  <Sparkles className="w-4 h-4" />
                  Join the Waitlist
                  <ArrowRight className="w-4 h-4" />
                </Link>
              )}
            </div>
          )}
          
          <form onSubmit={handleCreateAccount} className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-4 py-3 bg-slate-950 border border-slate-800 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-cyan-500"
                required
                disabled={loading}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">Password</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full px-4 py-3 bg-slate-950 border border-slate-800 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-cyan-500"
                required
                disabled={loading}
              />
            </div>
            
            {/* Terms Acceptance */}
            <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
              <label className="flex items-start gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={acceptedTerms}
                  onChange={(e) => setAcceptedTerms(e.target.checked)}
                  className="mt-1 w-5 h-5 rounded border-slate-500 text-cyan-500 focus:ring-cyan-500 bg-slate-900"
                  disabled={loading}
                />
                <span className="text-slate-300 text-sm leading-relaxed">
                  I am at least <strong className="text-white">18 years old</strong> and I have read and agree to the{' '}
                  <Link 
                    to="/legal" 
                    target="_blank" 
                    className="text-cyan-400 hover:text-cyan-300 underline inline-flex items-center gap-1"
                  >
                    Terms of Service
                    <ExternalLink className="w-3 h-3" />
                  </Link>
                  .
                </span>
              </label>
            </div>

            <button
              type="submit"
              disabled={loading || !acceptedTerms}
              className="w-full py-3 bg-cyan-500 hover:bg-cyan-400 text-white font-bold rounded-lg transition-colors shadow-[0_0_15px_rgba(6,182,212,0.3)] disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? 'Creating...' : 'Continue to Wallet Setup'}
              <ArrowRight className="w-4 h-4" />
            </button>
          </form>
          
          <p className="mt-6 text-center text-slate-400">
            Already have an account?{' '}
            <Link to="/login" className="text-cyan-400 hover:underline">Log in</Link>
          </p>
        </motion.div>
      </div>
    );
  }

  // Step 2: Wallet Choice (Create New or Import Existing)
  if (step === 'wallet') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-950 px-4 pt-20">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-md w-full bg-slate-900 p-8 rounded-2xl border border-slate-800 shadow-xl"
        >
          {/* Progress Indicator */}
          <div className="flex items-center justify-center gap-2 mb-8">
            <div className="w-8 h-8 rounded-full bg-green-500 text-white flex items-center justify-center text-sm font-bold">✓</div>
            <div className="w-12 h-0.5 bg-cyan-500"></div>
            <div className="w-8 h-8 rounded-full bg-cyan-500 text-white flex items-center justify-center text-sm font-bold">2</div>
            <div className="w-12 h-0.5 bg-slate-700"></div>
            <div className="w-8 h-8 rounded-full bg-slate-700 text-slate-500 flex items-center justify-center text-sm font-bold">3</div>
          </div>

          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-white mb-2">Connect Your Wallet</h2>
            <p className="text-slate-400">Step 2 of 3: Secure your NEURO earnings</p>
          </div>

          {error && <p className="text-red-300 text-sm mb-4 text-center bg-red-500/10 border border-red-500/50 rounded-lg p-3 shadow-[0_0_15px_rgba(239,68,68,0.15)]">{error}</p>}

          {/* Create New Wallet */}
          <button
            onClick={handleCreateWallet}
            disabled={loading}
            className="w-full py-4 bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-400 hover:to-purple-400 text-white font-bold rounded-lg transition-all shadow-[0_0_20px_rgba(6,182,212,0.3)] disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 mb-4"
          >
            {loading ? 'Generating...' : 'Create New Wallet'}
          </button>

          {/* Divider */}
          <div className="flex items-center gap-4 my-6">
            <div className="flex-1 h-px bg-slate-700"></div>
            <span className="text-slate-500 text-sm">or</span>
            <div className="flex-1 h-px bg-slate-700"></div>
          </div>

          {/* Import Existing */}
          <button
            onClick={() => setStep('import')}
            className="w-full py-4 bg-slate-800 hover:bg-slate-700 text-white font-bold rounded-lg transition-all border border-slate-600 flex items-center justify-center gap-2"
          >
            Import Existing Wallet
          </button>

          <p className="text-slate-500 text-xs text-center mt-4">
            Have a 12-word recovery phrase? Import your existing wallet.
          </p>
        </motion.div>
      </div>
    );
  }

  // Step 2b: Import Existing Wallet
  if (step === 'import') {
    const isValidFormat = importSecret.trim().split(/\s+/).length >= 12 || 
                          (importSecret.trim().length === 64 && /^[a-f0-9]+$/i.test(importSecret.trim()));
    
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-950 px-4 pt-20">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-md w-full bg-slate-900 p-8 rounded-2xl border border-slate-800 shadow-xl"
        >
          {/* Progress Indicator */}
          <div className="flex items-center justify-center gap-2 mb-8">
            <div className="w-8 h-8 rounded-full bg-green-500 text-white flex items-center justify-center text-sm font-bold">✓</div>
            <div className="w-12 h-0.5 bg-cyan-500"></div>
            <div className="w-8 h-8 rounded-full bg-cyan-500 text-white flex items-center justify-center text-sm font-bold">2</div>
            <div className="w-12 h-0.5 bg-slate-700"></div>
            <div className="w-8 h-8 rounded-full bg-slate-700 text-slate-500 flex items-center justify-center text-sm font-bold">3</div>
          </div>

          <div className="text-center mb-6">
            <h2 className="text-2xl font-bold text-white mb-2">Import Existing Wallet</h2>
            <p className="text-slate-400 text-sm">Enter your 12-word recovery phrase or secret token</p>
          </div>

          {error && <p className="text-red-300 text-sm mb-4 text-center bg-red-500/10 border border-red-500/50 rounded-lg p-3 shadow-[0_0_15px_rgba(239,68,68,0.15)]">{error}</p>}

          <div className="mb-6">
            <textarea
              value={importSecret}
              onChange={(e) => setImportSecret(e.target.value)}
              placeholder="Enter your 12-word recovery phrase or 64-character token..."
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-4 py-3 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500 resize-none font-mono text-sm h-32"
              autoComplete="off"
              spellCheck={false}
            />
            {importSecret && (
              <p className={`text-xs mt-2 ${isValidFormat ? 'text-green-400' : 'text-amber-400'}`}>
                {importSecret.trim().split(/\s+/).length >= 12 
                  ? `✓ Mnemonic detected (${importSecret.trim().split(/\s+/).length} words)`
                  : importSecret.trim().length === 64 
                    ? '✓ Token format detected'
                    : '⚠ Enter 12 words or 64-char token'}
              </p>
            )}
          </div>

          <button
            onClick={handleImportWallet}
            disabled={loading || !isValidFormat}
            className="w-full py-4 bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-400 hover:to-purple-400 text-white font-bold rounded-lg transition-all shadow-[0_0_20px_rgba(6,182,212,0.3)] disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 mb-4"
          >
            {loading ? 'Importing...' : 'Import Wallet'}
            <ArrowRight className="w-4 h-4" />
          </button>

          <button
            onClick={() => { setStep('wallet'); setError(''); setImportSecret(''); }}
            className="w-full py-3 text-slate-400 hover:text-white transition-colors text-sm"
          >
            ← Back to wallet options
          </button>
        </motion.div>
      </div>
    );
  }

  // Step 3: Mnemonic Display
  if (step === 'mnemonic' && wallet) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-950 px-4 pt-20 pb-12">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-2xl w-full bg-slate-900 p-8 rounded-2xl border border-slate-800 shadow-xl"
        >
          {/* Progress Indicator */}
          <div className="flex items-center justify-center gap-2 mb-8">
            <div className="w-8 h-8 rounded-full bg-green-500 text-white flex items-center justify-center text-sm font-bold">✓</div>
            <div className="w-12 h-0.5 bg-green-500"></div>
            <div className="w-8 h-8 rounded-full bg-green-500 text-white flex items-center justify-center text-sm font-bold">✓</div>
            <div className="w-12 h-0.5 bg-cyan-500"></div>
            <div className="w-8 h-8 rounded-full bg-cyan-500 text-white flex items-center justify-center text-sm font-bold">3</div>
          </div>

          <div className="text-center mb-6">
            <h2 className="text-3xl font-bold text-white mb-2">Wallet Created!</h2>
            <p className="text-slate-400">Step 3 of 3: Save your recovery phrase</p>
          </div>

          {/* CRITICAL WARNING */}
          <div className="bg-red-500/10 border-2 border-red-500/50 rounded-xl p-5 mb-6">
            <div className="flex items-start gap-3">
              <AlertTriangle className="text-red-400 flex-shrink-0 mt-0.5" size={24} />
              <div>
                <p className="text-red-300 font-bold text-lg mb-2">
                  ⚠️ SAVE THIS NOW - YOU WON'T SEE IT AGAIN
                </p>
                <ul className="text-red-200/80 text-xs space-y-1 list-disc list-inside">
                  <li>Write it down on paper and store it safely</li>
                  <li>Never share it with anyone - not even NeuroShard support</li>
                  <li>Anyone with this phrase can access your NEURO</li>
                  <li>We cannot recover your wallet if you lose this phrase</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Mnemonic Display */}
          <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 mb-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-white font-bold">Your 12-Word Recovery Phrase</h3>
              <button
                onClick={() => setShowMnemonic(!showMnemonic)}
                className="text-slate-400 hover:text-white transition-colors"
              >
                {showMnemonic ? <EyeOff size={20} /> : <Eye size={20} />}
              </button>
            </div>

            <div className={`grid grid-cols-2 sm:grid-cols-3 gap-3 mb-4 ${!showMnemonic ? 'blur-md select-none' : ''}`}>
              {wallet.mnemonic.split(' ').map((word: string, i: number) => (
                <div key={i} className="bg-slate-900 rounded-lg p-3 border border-slate-700">
                  <span className="text-slate-500 text-xs mr-2">{i + 1}.</span>
                  <span className="text-cyan-400 font-mono font-bold">{word}</span>
                </div>
              ))}
            </div>

            <button
              onClick={copyMnemonic}
              className="w-full bg-slate-700 hover:bg-slate-600 text-white font-medium py-3 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              {copiedMnemonic ? (
                <><Check size={18} className="text-green-400" /><span>Copied!</span></>
              ) : (
                <><Copy size={18} /><span>Copy Mnemonic</span></>
              )}
            </button>
          </div>

          {/* Secret Token for Runner */}
          <div className="bg-gradient-to-r from-cyan-900/30 to-purple-900/30 border border-cyan-500/30 rounded-xl p-6 mb-6">
            <div className="flex items-center gap-3 mb-3">
              <Terminal className="text-cyan-400" size={20} />
              <h3 className="text-white font-bold">Secret Token (For Node Runner)</h3>
            </div>
            <p className="text-slate-400 text-xs mb-4">
              Copy this token to paste into the NeuroShard Node Runner. It's derived from your mnemonic.
            </p>
            
            <div className="flex items-center gap-2 mb-3">
              <div className={`flex-1 bg-slate-900 border border-slate-700 rounded-lg p-3 font-mono text-xs break-all ${!showToken ? 'blur-sm select-none' : 'text-cyan-400'}`}>
                {wallet.token}
              </div>
              <button
                onClick={() => setShowToken(!showToken)}
                className="p-3 bg-slate-800 hover:bg-slate-700 text-slate-400 rounded-lg transition-colors"
              >
                {showToken ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
            
            <button
              onClick={copyToken}
              className="w-full bg-cyan-600 hover:bg-cyan-500 text-white font-medium py-3 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
            >
              {copiedToken ? (
                <><Check size={18} className="text-white" /><span>Token Copied!</span></>
              ) : (
                <><Copy size={18} /><span>Copy Token for Runner</span></>
              )}
            </button>
          </div>

          {/* Confirmation */}
          <div className="bg-slate-800 border border-slate-700 rounded-xl p-4 mb-6">
            <label className="flex items-start gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={confirmed}
                onChange={(e) => setConfirmed(e.target.checked)}
                className="mt-1 w-5 h-5 rounded border-slate-500 text-cyan-500 focus:ring-cyan-500"
              />
              <span className="text-slate-300 text-sm">
                I have <strong className="text-white">written down and securely stored</strong> my 12-word recovery phrase.
                I understand that <strong className="text-red-400">I cannot recover my wallet</strong> without it.
              </span>
            </label>
          </div>

          <button
            onClick={handleComplete}
            disabled={!confirmed}
            className="w-full py-4 bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-400 hover:to-purple-400 text-white font-bold rounded-lg transition-all shadow-[0_0_20px_rgba(6,182,212,0.3)] disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            Complete Registration & Go to Dashboard
            <ArrowRight className="w-4 h-4" />
          </button>
        </motion.div>
      </div>
    );
  }

  return null;
};
