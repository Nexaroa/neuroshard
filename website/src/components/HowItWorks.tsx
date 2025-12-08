import { motion } from 'framer-motion';

export const HowItWorks = () => {
  return (
    <section id="concept" className="py-24 bg-gradient-to-b from-slate-950 to-slate-900 overflow-hidden">
      <div className="container mx-auto px-6">
        <div className="flex flex-col lg:flex-row items-center gap-16">
          <div className="lg:w-1/2">
            <motion.h2
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="text-3xl md:text-4xl font-bold text-white mb-6"
            >
              The Global Relay Race
            </motion.h2>
            <motion.p
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
              className="text-slate-300 text-lg mb-8 leading-relaxed"
            >
              Instead of one massive server holding the entire model, NeuroShard distributes layers across the network. Intelligence emerges from the collective relay of tokens.
            </motion.p>
            
            <ul className="space-y-6">
              {[
                { title: 'Model Sharding', desc: 'Node A loads Layers 0-4. Node B loads Layers 4-8. Memory requirements are slashed, enabling consumer hardware participation.' },
                { title: 'Token Relay', desc: 'Activations are quantized, compressed, and relayed via gRPC. "Session Affinity" ensures consistent routing for caching.' },
                { title: 'Speculative Execution', desc: 'Clients generate draft tokens locally. The network verifies them in parallel, hiding internet latency.' },
                { title: 'Proof of Neural Work', desc: 'Nodes are rewarded for verified computation. Audits ensure trust without redundant execution, creating a fair economy.' }
              ].map((item, idx) => (
                <motion.li
                  key={idx}
                  initial={{ opacity: 0, y: 10 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: 0.3 + (idx * 0.1) }}
                  className="flex gap-4"
                >
                  <span className="flex-shrink-0 w-8 h-8 rounded-full bg-cyan-500/20 text-cyan-400 flex items-center justify-center font-bold border border-cyan-500/30">
                    {idx + 1}
                  </span>
                  <div>
                    <h4 className="text-white font-bold mb-1">{item.title}</h4>
                    <p className="text-slate-400">{item.desc}</p>
                  </div>
                </motion.li>
              ))}
            </ul>
          </div>

          <div className="lg:w-1/2 relative">
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              className="relative z-10 bg-slate-800/50 backdrop-blur rounded-2xl border border-slate-700 p-8 shadow-2xl"
            >
              {/* Visual representation of nodes */}
              <div className="flex flex-col md:flex-row justify-between items-center mb-8 relative gap-8 md:gap-0">
                {/* Connecting Line (Desktop) */}
                <div className="hidden md:block absolute top-1/2 left-0 w-full h-1 bg-slate-700 -z-10"></div>
                {/* Connecting Line (Mobile) */}
                <div className="md:hidden absolute left-1/2 top-0 w-1 h-full bg-slate-700 -z-10 -translate-x-1/2"></div>
                
                <motion.div
                   animate={{ backgroundPosition: ["0% 0%", "100% 0%"] }}
                   className="hidden md:block absolute top-1/2 left-0 w-full h-1 bg-gradient-to-r from-transparent via-cyan-500 to-transparent -z-10 opacity-50"
                ></motion.div>

                {/* Nodes */}
                {['Alice', 'Bob', 'Charlie'].map((name, i) => (
                  <div key={name} className="flex flex-col items-center gap-2 bg-slate-900 p-4 rounded-xl border border-slate-600 z-10 min-w-[100px] w-full md:w-auto">
                     <div className={`w-3 h-3 rounded-full ${i === 1 ? 'bg-cyan-500 animate-pulse' : 'bg-slate-500'}`}></div>
                     <span className="font-mono text-xs text-slate-400">Node {name}</span>
                     <span className="text-xs text-cyan-400 font-bold">Layers {i*4}-{(i+1)*4}</span>
                  </div>
                ))}
              </div>
              
              <div className="font-mono text-xs md:text-sm bg-black/50 p-4 rounded border border-slate-800 text-green-400 overflow-x-auto">
                <p className="opacity-50 whitespace-nowrap">{`> Session ID: 8f92-a3b1...`}</p>
                <p className="whitespace-nowrap">{`> Node Alice: Processing Batch (Draft K=5)...`}</p>
                <p className="whitespace-nowrap">{`> Node Alice: Forward Pass Layers 0-4 [OK]`}</p>
                <p className="text-cyan-400 whitespace-nowrap">{`> Relaying 0.37MB compressed tensor to Bob...`}</p>
                <p className="animate-pulse whitespace-nowrap">{`> Node Bob: Receiving...`}</p>
              </div>
            </motion.div>
            
            {/* Decor */}
            <div className="absolute -top-10 -right-10 w-64 h-64 bg-cyan-500/20 rounded-full blur-3xl -z-0"></div>
            <div className="absolute -bottom-10 -left-10 w-64 h-64 bg-purple-500/20 rounded-full blur-3xl -z-0"></div>
          </div>
        </div>
      </div>
    </section>
  );
};
