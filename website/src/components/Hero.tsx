import { motion } from 'framer-motion';
import { Sparkles, FileText, Activity, ArrowRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { NetworkBackground } from './NetworkBackground';
import { LiveStats } from './LiveStats';

export const Hero = () => {
  const navigate = useNavigate();

  return (
    <section className="relative min-h-screen flex items-center justify-center pt-20 overflow-hidden bg-slate-950">
      {/* Overlay */}
      <div className="absolute inset-0 bg-slate-950/80 bg-gradient-to-b from-slate-950/90 via-slate-950/80 to-slate-950"></div>
      
      {/* Animated Network Background */}
      <NetworkBackground />

      <div className="container mx-auto px-6 relative z-10 text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <div className="inline-flex items-center gap-2 py-1 px-3 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-sm font-medium mb-6">
            <Activity className="w-4 h-4" />
            Limited Spots Available
          </div>
          <h1 className="text-5xl md:text-7xl font-extrabold text-white tracking-tight mb-6 leading-tight">
            The Global Brain. <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-600">
              Powered by Everyone.
            </span>
          </h1>
          <p className="text-xl md:text-2xl text-slate-300 max-w-2xl mx-auto mb-10 leading-relaxed">
            Join the distributed AI revolution. Register your hardware, reserve your mining node, and earn NEURO tokens.
          </p>
          
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => navigate('/join')}
              className="group w-full sm:w-auto px-8 py-4 bg-cyan-500 hover:bg-cyan-400 text-white rounded-full font-bold text-lg flex items-center justify-center gap-2 shadow-[0_0_30px_rgba(6,182,212,0.4)] transition-all cursor-pointer"
            >
              <Sparkles className="w-5 h-5" />
              Reserve Your Node
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => navigate('/whitepaper')}
              className="w-full sm:w-auto px-8 py-4 bg-slate-800 hover:bg-slate-700 text-white rounded-full font-bold text-lg flex items-center justify-center gap-2 transition-all border border-slate-700 cursor-pointer"
            >
              <FileText className="w-5 h-5" />
              Read Whitepaper
            </motion.button>
          </div>
        </motion.div>

        {/* Live Stats Section */}
        <LiveStats />

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1, duration: 1 }}
          className="mt-12 flex flex-col sm:flex-row justify-center gap-4 sm:gap-8 text-slate-500 text-sm uppercase tracking-widest font-semibold"
        >
          <div className="flex items-center justify-center gap-2">
            <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></span>
            100% Decentralized
          </div>
          <div className="flex items-center justify-center gap-2">
            <span className="w-2 h-2 bg-cyan-500 rounded-full animate-pulse"></span>
            Limitless Scale
          </div>
          <div className="flex items-center justify-center gap-2">
            <span className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></span>
            Pure Intelligence
          </div>
        </motion.div>
      </div>
    </section>
  );
};
