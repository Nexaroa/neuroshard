import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Send, Zap, Terminal, Server, Lock, Activity, LayoutDashboard, Coins } from 'lucide-react';
import { API_URL } from '../config/api';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate, Link } from 'react-router-dom';

export const Chat = () => {
  const { user, token, refreshUser } = useAuth();
  const navigate = useNavigate();
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Array<{ role: 'user' | 'assistant'; content: string }>>([
    { role: 'assistant', content: 'Hello! I am NeuroShard, a distributed AI running across a global swarm of consumer devices. How can I help you today?' }
  ]);
  const [loading, setLoading] = useState(false);
  const [nodeStatus, setNodeStatus] = useState(false);
  const [neuroBalance, setNeuroBalance] = useState<number | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!user) {
      // Optional: Redirect if not logged in, or show a lock screen
      // navigate('/login');
    } else {
      // Check status initially
      checkNodeStatus();
      fetchNeuroBalance();
    }
  }, [user, navigate]);

  const fetchNeuroBalance = async () => {
    if (!user?.node_id) return;
    try {
      const response = await axios.get(`${API_URL}/api/node/neuro`, {
        params: { node_id: user.node_id }
      });
      setNeuroBalance(response.data.neuro_balance || 0);
    } catch (error) {
      console.error("Failed to fetch NEURO balance", error);
      setNeuroBalance(0);
    }
  };

  const checkNodeStatus = async () => {
    if (!token) return;
    try {
      const res = await axios.get(`${API_URL}/api/users/me/node_status`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setNodeStatus(res.data.active);
    } catch (e) {
      console.error(e);
    }
  };

  const scrollToBottom = () => {
    // Only scroll the chat container, not the whole window
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Scroll to top on mount
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    if (!user) {
      navigate('/login');
      return;
    }

    const userMsg = input.trim();
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setInput('');
    setLoading(true);

    try {
      // Call our backend proxy which forwards to the swarm
      const response = await axios.post(`${API_URL}/api/chat`, {
        prompt: userMsg,
        max_new_tokens: 50
      }, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });

      const generatedText = response.data.text || response.data.result || "No response generated.";
      
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: generatedText
      }]);

      // Refresh user data and NEURO balance after successful chat
      refreshUser();
      fetchNeuroBalance();

      } catch (err: any) {
      let errorMsg = "⚠️ Error: Could not connect to the swarm. Please try again later.";
      if (err.response && err.response.status === 401) {
        errorMsg = "⚠️ Session expired. Please login again.";
        // logout();
      }
      if (err.response && err.response.status === 402) {
        errorMsg = `💰 Insufficient NEURO: ${err.response.data.detail}`;
      }
      if (err.response && err.response.status === 403) {
        errorMsg = "🔒 Access Restricted: You must have an active NeuroShard Node running to use the chat. Please start your node with your Node Token.";
      }
      setMessages(prev => [...prev, { 
        role: 'assistant',
        content: errorMsg
      }]);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  if (!user) {
    return (
      <section className="pt-32 pb-24 min-h-screen bg-slate-950 relative overflow-hidden flex items-center justify-center">
        <div className="text-center p-8 max-w-md">
          <div className="inline-flex p-4 bg-slate-900 rounded-full mb-6 border border-slate-800">
            <Lock className="w-8 h-8 text-cyan-400" />
          </div>
          <h2 className="text-3xl font-bold text-white mb-4">Access Restricted</h2>
          <p className="text-slate-400 mb-8">
            The NeuroShard Swarm Chat is currently in closed beta. 
            Only active node operators who have contributed to the network can access the live model.
          </p>
          <div className="flex flex-col gap-4">
            <button 
              onClick={() => navigate('/login')}
              className="w-full py-3 bg-cyan-500 hover:bg-cyan-400 text-white font-bold rounded-xl transition-colors"
            >
              Log In
            </button>
            <button 
              onClick={() => navigate('/signup')}
              className="w-full py-3 bg-slate-800 hover:bg-slate-700 text-white font-medium rounded-xl transition-colors border border-slate-700"
            >
              Create Account & Join Swarm
            </button>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="pt-28 pb-6 h-screen flex flex-col bg-slate-950 relative overflow-hidden">
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden z-0 pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/5 rounded-full blur-[120px]"></div>
        <div className="absolute bottom-1/3 right-1/4 w-96 h-96 bg-purple-500/5 rounded-full blur-[120px]"></div>
      </div>

      <div className="container mx-auto px-4 lg:px-6 relative z-10 flex-1 flex gap-6 min-h-0">

        {/* Sidebar - User Info */}
        <div className="w-72 hidden lg:flex flex-col gap-4 overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-slate-800">
          {/* User Card */}
          <div className="bg-slate-900/60 border border-slate-800/60 rounded-2xl p-5 backdrop-blur-sm">
            <div className="mb-4">
                <p className="text-xs text-slate-400 font-medium uppercase tracking-wider">Logged in as</p>
                <p className="font-medium text-white truncate text-sm">{user.email}</p>
            </div>

            <div className="bg-slate-950/80 rounded-xl p-4 border border-slate-800 mb-4">
              <p className="text-xs text-slate-400 mb-1 flex items-center gap-2 font-medium">
                <Coins className="w-3 h-3 text-yellow-400" />
                NEURO Balance
              </p>
              <p className="text-2xl font-bold text-white tracking-tight">
                {neuroBalance !== null ? neuroBalance.toFixed(4) : '...'}
              </p>
            </div>

            <div className="flex items-center gap-3 mb-4 px-1">
              <div className={`w-2.5 h-2.5 rounded-full ${nodeStatus ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
              <span className={`text-sm font-medium ${nodeStatus ? 'text-green-400' : 'text-red-400'}`}>
                {nodeStatus ? 'Node Active' : 'Node Offline'}
              </span>
            </div>

            <Link to="/dashboard" className="flex items-center justify-center gap-2 w-full py-2.5 bg-slate-800 hover:bg-slate-700 text-white rounded-xl transition-colors border border-slate-700 font-medium text-sm">
              <LayoutDashboard className="w-4 h-4" />
              Manage Node
            </Link>
          </div>

          {/* Info Card */}
          <div className="bg-slate-900/60 border border-slate-800/60 rounded-2xl p-5 backdrop-blur-sm flex-1">
            <h3 className="font-bold text-white mb-4 flex items-center gap-2 text-sm">
              <Activity className="w-4 h-4 text-cyan-400" />
              Network Stats
            </h3>
            <p className="text-slate-400 text-xs mb-4 leading-relaxed">
              Pricing: 0.1 NEURO per 1M tokens
            </p>
            <div className="space-y-3">
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">Uptime Reward</span>
                <span className="text-slate-300">0.0005 NEURO/min</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">Training Reward</span>
                <span className="text-slate-300">0.0001 NEURO/batch</span>
              </div>
            </div>
            <div className="mt-6 pt-6 border-t border-slate-800/60">
              <p className="text-[10px] text-slate-500 text-center leading-normal">
                Keep your node running to earn NEURO tokens.
              </p>
            </div>
          </div>
        </div>

        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col bg-slate-900/80 border border-slate-800/60 rounded-2xl backdrop-blur-md overflow-hidden shadow-2xl">
          {/* Header */}
          <div className="px-6 py-4 border-b border-slate-800/60 bg-slate-950/30 flex items-center justify-between flex-shrink-0">
            <div>
              <h1 className="font-bold text-white text-lg leading-none mb-1">Swarm Chat</h1>
              <p className="text-slate-400 text-xs flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-green-500"></span>
                Distributed Intelligence Online
              </p>
            </div>
            <div className="hidden sm:block text-xs text-slate-500 bg-slate-900 py-1 px-3 rounded-full border border-slate-800">
              Model: NeuroLLM-v1 (Decentralized)
            </div>
          </div>

          {/* Mobile Stats Bar */}
          <div className="lg:hidden px-6 py-2 bg-slate-900/40 border-b border-slate-800/60 flex items-center justify-between text-xs backdrop-blur-md">
             <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${nodeStatus ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
                <span className={nodeStatus ? 'text-green-400 font-medium' : 'text-slate-400 font-medium'}>{nodeStatus ? 'Node Active' : 'Offline'}</span>
             </div>
             <div className="flex items-center gap-1.5 bg-slate-950/50 px-2 py-1 rounded-lg border border-slate-800/50">
                <Coins className="w-3 h-3 text-yellow-400" />
                <span className="text-white font-mono font-medium">{neuroBalance !== null ? neuroBalance.toFixed(2) : '0.00'}</span>
             </div>
          </div>

          {/* Chat Window */}
          <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
            {messages.map((msg, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                    className={`max-w-[85%] sm:max-w-[75%] p-4 rounded-2xl shadow-sm ${msg.role === 'user'
                      ? 'bg-cyan-600 text-white rounded-tr-sm shadow-cyan-900/20'
                      : 'bg-slate-800 text-slate-200 rounded-tl-sm border border-slate-700/50'
                  }`}
                >
                  {msg.role === 'assistant' && (
                     <div className="flex items-center gap-2 mb-2 text-[10px] text-cyan-400 font-mono uppercase tracking-wider font-bold">
                        <Server className="w-3 h-3" />
                        NeuroShard Node
                     </div>
                  )}
                  <p className="leading-relaxed whitespace-pre-wrap text-sm sm:text-base">{msg.content}</p>
                </div>
              </motion.div>
            ))}
            {loading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex justify-start"
              >
                <div className="bg-slate-800 p-4 rounded-2xl rounded-tl-sm border border-slate-700/50">
                  <div className="flex items-center gap-1.5">
                    <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              </motion.div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="p-4 sm:p-5 bg-slate-950/30 border-t border-slate-800/60 flex-shrink-0">
            <form onSubmit={handleSubmit} className="relative max-w-4xl mx-auto w-full">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask the swarm anything..."
                className="w-full bg-slate-900 border border-slate-800 rounded-xl py-4 pl-5 pr-14 text-white placeholder:text-slate-500 focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/50 transition-all shadow-inner"
                disabled={loading}
              />
              <button
                type="submit"
                disabled={loading || !input.trim()}
                className="absolute right-2 top-2 p-2 bg-cyan-500 hover:bg-cyan-400 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-cyan-500/20"
              >
                {loading ? <Zap className="w-5 h-5 animate-pulse" /> : <Send className="w-5 h-5" />}
              </button>
            </form>
            <div className="mt-3 text-center">
               <p className="text-[10px] text-slate-500">
                  <Terminal className="w-3 h-3 inline mr-1.5" />
                  Powered by NeuroLLM (Decentralized) • Early model - quality improves as network grows!
               </p>
              </div>
            </div>
          </div>
        </div>
    </section>
  );
};
