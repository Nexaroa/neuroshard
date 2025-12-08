import { motion } from 'framer-motion';
import { Cpu, Zap, Shield, Layers } from 'lucide-react';

const technologies = [
  {
    icon: <Cpu className="w-8 h-8 text-blue-400" />,
    title: 'Speculative Decoding',
    description: 'Clients generate draft tokens locally, which are verified in batches by the network. This "hides" network latency, making distributed inference feel real-time.',
  },
  {
    icon: <Layers className="w-8 h-8 text-purple-400" />,
    title: 'Smart Sharding & Caching',
    description: 'Nodes maintain KV caches for active sessions ("Session Affinity"). Only new tokens are transmitted, reducing complexity from O(N²) to O(N).',
  },
  {
    icon: <Zap className="w-8 h-8 text-yellow-400" />,
    title: '8x Bandwidth Reduction',
    description: 'Activations are quantized to INT8 and compressed with Zlib before transmission via gRPC/Protobuf, enabling high-speed relay over consumer internet.',
  },
  {
    icon: <Shield className="w-8 h-8 text-green-400" />,
    title: 'Proof of Neural Work',
    description: 'A revolutionary consensus mechanism that rewards useful computation (inference & training) instead of idle uptime. Active nodes earn significantly more, creating a robust "Give-to-Get" economy.',
  },
];

export const Technology = () => {
  return (
    <section id="technology" className="py-24 bg-slate-900 relative overflow-hidden">
      {/* Background Elements */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden z-0 pointer-events-none">
        <div className="absolute -right-20 top-40 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl"></div>
        <div className="absolute -left-20 bottom-40 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl"></div>
      </div>

      <div className="container mx-auto px-6 relative z-10">
        <div className="text-center mb-16">
          <motion.h2 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-3xl md:text-5xl font-bold text-white mb-6"
          >
            Powered by <span className="text-cyan-400">NeuroShard</span> Protocol
          </motion.h2>
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
            className="text-slate-400 text-lg max-w-2xl mx-auto"
          >
            We've solved the latency and bandwidth challenges of distributed AI through a novel combination of pipeline parallelism and optimistic consensus.
          </motion.p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {technologies.map((tech, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: index % 2 === 0 ? -30 : 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1, duration: 0.5 }}
              className="flex flex-col sm:flex-row gap-6 p-6 rounded-2xl bg-slate-950/50 border border-slate-800 hover:border-cyan-500/30 transition-colors"
            >
              <div className="flex-shrink-0 self-start">
                <div className="p-3 bg-slate-900 rounded-xl border border-slate-800 shadow-lg">
                  {tech.icon}
                </div>
              </div>
              <div>
                <h3 className="text-xl font-bold text-white mb-2">{tech.title}</h3>
                <p className="text-slate-400 leading-relaxed">
                  {tech.description}
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

