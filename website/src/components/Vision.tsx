import { motion } from 'framer-motion';
import { Globe, Coins, Brain, ArrowRight, Sparkles } from 'lucide-react';

export const Vision = () => {
  return (
    <section className="py-32 bg-slate-950 relative overflow-hidden">
      {/* Animated gradient orbs */}
      <div className="absolute top-20 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-[120px] animate-pulse"></div>
      <div className="absolute bottom-20 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-[120px] animate-pulse" style={{ animationDelay: '1s' }}></div>
      
      <div className="container mx-auto px-6 relative z-10">
        {/* Section Header */}
        <motion.div 
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-20"
        >
          <div className="inline-flex items-center gap-2 py-1 px-4 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-400 text-sm font-medium mb-6">
            <Sparkles className="w-4 h-4" />
            The Vision
          </div>
          <h2 className="text-4xl md:text-6xl font-bold text-white mb-6 leading-tight">
            What If AI Belonged <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500">
              To Everyone?
            </span>
          </h2>
          <p className="text-xl text-slate-400 max-w-3xl mx-auto leading-relaxed">
            Today's AI is locked in corporate data centers. We're building something different - 
            a <span className="text-white font-semibold">globally distributed intelligence</span> that 
            anyone can contribute to, and everyone can benefit from.
          </p>
        </motion.div>

        {/* The Problem vs Solution */}
        <div className="grid lg:grid-cols-2 gap-8 mb-24">
          {/* The Problem */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="relative"
          >
            <div className="absolute inset-0 bg-gradient-to-br from-red-500/5 to-orange-500/5 rounded-3xl"></div>
            <div className="relative bg-slate-900/80 backdrop-blur border border-red-500/20 rounded-3xl p-8 md:p-10 h-full">
              <div className="inline-flex items-center gap-2 py-1 px-3 rounded-full bg-red-500/10 text-red-400 text-xs font-bold uppercase tracking-wider mb-6">
                The Problem
              </div>
              <h3 className="text-2xl md:text-3xl font-bold text-white mb-6">
                AI Is Centralized & Fragile
              </h3>
              <ul className="space-y-4 text-slate-300">
                <li className="flex items-start gap-3">
                  <span className="text-red-400 mt-1">✗</span>
                  <span>Trillion-dollar models locked behind corporate APIs</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-red-400 mt-1">✗</span>
                  <span>Massive GPU clusters consume as much power as small cities</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-red-400 mt-1">✗</span>
                  <span>Single points of failure - if OpenAI goes down, millions are affected</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-red-400 mt-1">✗</span>
                  <span>You pay to use it, but never own any of it</span>
                </li>
              </ul>
            </div>
          </motion.div>

          {/* The Solution */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="relative"
          >
            <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-green-500/5 rounded-3xl"></div>
            <div className="relative bg-slate-900/80 backdrop-blur border border-cyan-500/20 rounded-3xl p-8 md:p-10 h-full">
              <div className="inline-flex items-center gap-2 py-1 px-3 rounded-full bg-cyan-500/10 text-cyan-400 text-xs font-bold uppercase tracking-wider mb-6">
                Our Solution
              </div>
              <h3 className="text-2xl md:text-3xl font-bold text-white mb-6">
                A Global Brain, Owned by All
              </h3>
              <ul className="space-y-4 text-slate-300">
                <li className="flex items-start gap-3">
                  <span className="text-cyan-400 mt-1">✓</span>
                  <span>Run AI on millions of devices worldwide - laptops, servers, phones</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-cyan-400 mt-1">✓</span>
                  <span>No single point of failure - the network is the computer</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-cyan-400 mt-1">✓</span>
                  <span>Contribute compute, earn NEURO tokens - real ownership</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-cyan-400 mt-1">✓</span>
                  <span>The model learns continuously from collective training</span>
                </li>
              </ul>
            </div>
          </motion.div>
        </div>

        {/* The Three Pillars */}
        <motion.div 
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h3 className="text-2xl md:text-3xl font-bold text-white mb-4">
            Three Pillars of the New AI Economy
          </h3>
          <p className="text-slate-400 max-w-2xl mx-auto">
            NeuroShard isn't just distributed computing - it's a new paradigm where participation creates value.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-6">
          {[
            {
              icon: <Brain className="w-10 h-10" />,
              title: "Contribute Intelligence",
              description: "Run a node. Your device becomes part of a global neural network. Even a laptop can hold layers of the model and process tokens.",
              gradient: "from-purple-500 to-pink-500",
              borderColor: "border-purple-500/30",
            },
            {
              icon: <Coins className="w-10 h-10" />,
              title: "Earn NEURO Tokens",
              description: "Every forward pass, every gradient computed - you get paid. Proof of Neural Work ensures only real work is rewarded.",
              gradient: "from-cyan-500 to-blue-500",
              borderColor: "border-cyan-500/30",
            },
            {
              icon: <Globe className="w-10 h-10" />,
              title: "Shape the Future",
              description: "This isn't someone else's AI. Stake NEURO to vote on model upgrades. The community decides how the network evolves.",
              gradient: "from-green-500 to-emerald-500",
              borderColor: "border-green-500/30",
            },
          ].map((pillar, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.15 }}
              className={`relative group`}
            >
              <div className={`absolute inset-0 bg-gradient-to-br ${pillar.gradient} opacity-0 group-hover:opacity-10 rounded-3xl transition-opacity duration-500`}></div>
              <div className={`relative bg-slate-900/50 backdrop-blur border ${pillar.borderColor} rounded-3xl p-8 h-full hover:border-opacity-60 transition-all duration-300`}>
                <div className={`inline-flex p-4 rounded-2xl bg-gradient-to-br ${pillar.gradient} text-white mb-6 shadow-lg`}>
                  {pillar.icon}
                </div>
                <h4 className="text-xl font-bold text-white mb-3">{pillar.title}</h4>
                <p className="text-slate-400 leading-relaxed">{pillar.description}</p>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Call to Action */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mt-20 text-center"
        >
          <div className="inline-flex flex-col sm:flex-row items-center gap-4">
            <a
              href="/signup"
              className="group px-8 py-4 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white rounded-full font-bold text-lg flex items-center gap-3 shadow-[0_0_30px_rgba(6,182,212,0.4)] transition-all"
            >
              Join the Network
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </a>
            <span className="text-slate-500">
              Free to join • Earn from day one
            </span>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

