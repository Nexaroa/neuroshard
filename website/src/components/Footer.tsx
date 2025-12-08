import { Link } from 'react-router-dom';
import { Github, Twitter, FileText, ExternalLink, MessageCircle } from 'lucide-react';
import logo from '../assets/logo_white.png';

export const Footer = () => {
  return (
    <footer className="bg-slate-950 border-t border-slate-800/50 hidden md:block">
      {/* Main Footer */}
      <div className="container mx-auto px-6 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-10 md:gap-8">
          {/* Brand */}
          <div className="md:col-span-2">
            <div className="flex items-center gap-2 text-xl font-bold text-white mb-3">
              <img src={logo} alt="NeuroShard Logo" className="h-7 w-auto" />
              <span>Neuro<span className="text-cyan-400">Shard</span></span>
            </div>
            <p className="text-slate-400 text-sm leading-relaxed max-w-sm">
              Decentralized AI training network. Contributing compute, 
              earning NEURO, building collective intelligence.
            </p>
            {/* Social Icons */}
            <div className="flex items-center gap-3 mt-5">
              <a 
                href="https://github.com/Nexaroa/neuroshard" 
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 bg-slate-800/50 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-white transition-all"
                aria-label="GitHub"
              >
                <Github className="w-4 h-4" />
              </a>
              <a 
                href="https://x.com/shardneuro" 
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 bg-slate-800/50 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-white transition-all"
                aria-label="X (Twitter)"
              >
                <Twitter className="w-4 h-4" />
              </a>
              <a 
                href="https://discord.gg/4R49xpj7vn" 
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 bg-slate-800/50 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-white transition-all"
                aria-label="Discord"
              >
                <MessageCircle className="w-4 h-4" />
              </a>
              <a 
                href="https://docs.neuroshard.com" 
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 bg-slate-800/50 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-white transition-all"
                aria-label="Documentation"
              >
                <FileText className="w-4 h-4" />
              </a>
            </div>
          </div>

          {/* Resources */}
          <div>
            <h4 className="text-white font-semibold text-sm mb-4">Resources</h4>
            <ul className="space-y-2.5">
              <li>
                <Link to="/whitepaper" className="text-slate-400 hover:text-cyan-400 text-sm transition-colors inline-flex items-center gap-1">
                  Whitepaper
                </Link>
              </li>
              <li>
                <a href="https://docs.neuroshard.com" target="_blank" rel="noopener noreferrer" className="text-slate-400 hover:text-cyan-400 text-sm transition-colors inline-flex items-center gap-1">
                  Documentation
                  <ExternalLink className="w-3 h-3 opacity-50" />
                </a>
              </li>
              <li>
                <Link to="/download" className="text-slate-400 hover:text-cyan-400 text-sm transition-colors">
                  Download Node
                </Link>
              </li>
              <li>
                <Link to="/ledger" className="text-slate-400 hover:text-cyan-400 text-sm transition-colors">
                  Ledger Explorer
                </Link>
              </li>
            </ul>
          </div>

          {/* Legal */}
          <div>
            <h4 className="text-white font-semibold text-sm mb-4">Legal</h4>
            <ul className="space-y-2.5">
              <li>
                <Link to="/legal" className="text-slate-400 hover:text-cyan-400 text-sm transition-colors">
                  Terms of Service
                </Link>
              </li>
              <li>
                <Link to="/legal" className="text-slate-400 hover:text-cyan-400 text-sm transition-colors">
                  Privacy Policy
                </Link>
              </li>
              <li>
                <Link to="/legal" className="text-slate-400 hover:text-cyan-400 text-sm transition-colors">
                  Token Disclaimer
                </Link>
              </li>
              <li>
                <Link to="/legal" className="text-slate-400 hover:text-cyan-400 text-sm transition-colors">
                  Risk Disclosure
                </Link>
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* Bottom Bar */}
      <div className="border-t border-slate-800/50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex flex-col md:flex-row justify-between items-center gap-3">
            <p className="text-slate-500 text-xs">
              © {new Date().getFullYear()} Nexaroa · Apache 2.0
            </p>
            <p className="text-slate-600 text-xs text-center md:text-right max-w-xl">
              NEURO is a utility token, not an investment. AI outputs may be inaccurate. 
              <Link to="/legal" className="text-slate-500 hover:text-cyan-400 ml-1">
                Full disclaimer →
              </Link>
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
};
