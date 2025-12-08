import { motion } from 'framer-motion';
import { Sparkles, Zap, Laptop, ArrowRight, Users, TrendingUp } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export const Download = () => {
  const navigate = useNavigate();

  return (
    <section id="download" className="py-24 bg-slate-950">
      <div className="container mx-auto px-6">
        <div className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-3xl p-8 md:p-16 border border-slate-700 relative overflow-hidden text-center">
          
          <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-cyan-500 via-blue-500 to-cyan-400"></div>

          <div className="relative z-10 max-w-3xl mx-auto">
            <div className="inline-flex items-center gap-2 py-1.5 px-4 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-sm font-medium mb-6">
              <Users className="w-4 h-4" />
              Limited Spots • Join the Waitlist
            </div>
            
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">
              Reserve Your Node
            </h2>
            <p className="text-slate-300 text-lg mb-10">
              Register your hardware and secure your spot in the NeuroShard network. Get your estimated earnings and a unique referral link to boost your priority.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12 max-w-4xl mx-auto">
              <div className="bg-slate-950/50 p-6 rounded-xl border border-slate-700 hover:border-cyan-500/30 transition-colors">
                <Laptop className="w-8 h-8 text-cyan-400 mx-auto mb-4" />
                <h3 className="text-white font-bold mb-2">Register Hardware</h3>
                <p className="text-sm text-slate-400">Tell us your GPU, RAM, and internet specs.</p>
              </div>
              <div className="bg-slate-950/50 p-6 rounded-xl border border-slate-700 hover:border-emerald-500/30 transition-colors">
                <TrendingUp className="w-8 h-8 text-emerald-400 mx-auto mb-4" />
                <h3 className="text-white font-bold mb-2">See Your Earnings</h3>
                <p className="text-sm text-slate-400">Get estimated daily NEURO based on your setup.</p>
              </div>
              <div className="bg-slate-950/50 p-6 rounded-xl border border-slate-700 hover:border-blue-500/30 transition-colors">
                <Zap className="w-8 h-8 text-blue-400 mx-auto mb-4" />
                <h3 className="text-white font-bold mb-2">Get Your Neuro Link</h3>
                <p className="text-sm text-slate-400">Share to boost priority & earn referral bonuses.</p>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="group w-full sm:w-auto bg-cyan-500 hover:bg-cyan-400 text-white px-10 py-4 rounded-full font-bold text-lg shadow-[0_0_30px_rgba(6,182,212,0.4)] transition-all flex items-center justify-center gap-2"
                onClick={() => navigate('/join')}
              >
                <Sparkles className="w-5 h-5" />
                Join the Waitlist
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </motion.button>
            </div>
            
            <p className="mt-6 text-sm text-slate-500">
              Works on Windows, macOS, and Linux • CPU, NVIDIA GPU, or Apple Silicon
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};
