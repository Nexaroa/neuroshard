import { useState, useEffect } from 'react';
import { Menu, X, LogOut, Shield, Zap, ChevronDown, LayoutDashboard, Download, Hexagon } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import logo from '../assets/logo_white.png';

export const Header = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const { user, logout, isLoading } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleLogout = () => {
    logout();
    navigate('/');
    setIsProfileOpen(false);
  };

  const navLinks = [
    { name: 'Homepage', href: '/' },
    { name: 'Training', href: '/training' },
    { name: 'Ledger Explorer', href: '/ledger' },
    { name: 'Whitepaper', href: '/whitepaper' },
  ];

  return (
    <header
      className={`fixed top-0 w-full z-50 transition-all duration-300 ${
        scrolled ? 'bg-slate-950/80 backdrop-blur-md py-4 shadow-lg' : 'bg-transparent py-6'
      }`}
    >
      <div className="container mx-auto px-6 flex justify-between items-center">
        <Link to="/" className="flex items-center gap-2 text-2xl font-bold text-white">
          <img src={logo} alt="NeuroShard Logo" className="h-10 w-auto" />
          <span className="tracking-tight">Neuro<span className="text-cyan-400">Shard</span></span>
        </Link>

        {/* Desktop Nav */}
        <nav className="hidden md:flex items-center gap-8">
          {navLinks.map((link) => (
            <a
              key={link.name}
              href={link.href}
              className="text-sm font-medium text-slate-300 hover:text-cyan-400 transition-colors"
            >
              {link.name}
            </a>
          ))}
          
          {/* User section */}
          <div className="flex items-center gap-4 justify-end">
            {isLoading ? (
              <div className="flex items-center gap-4 w-full justify-end">
                <div className="h-8 w-24 bg-slate-800 animate-pulse rounded"></div>
              </div>
            ) : user ? (
              <div className="flex items-center gap-4">
                <Link 
                  to="/chat" 
                  className="bg-cyan-500 hover:bg-cyan-400 text-white px-4 py-2 rounded-full text-sm font-semibold transition-all shadow-[0_0_15px_rgba(6,182,212,0.3)] hover:shadow-[0_0_25px_rgba(6,182,212,0.5)] whitespace-nowrap flex items-center gap-2"
                >
                  <Zap className="w-4 h-4 fill-current" />
                  Live Demo
                </Link>

                {/* User Dropdown */}
                <div className="relative">
                  <button 
                    onClick={() => setIsProfileOpen(!isProfileOpen)}
                    className="flex items-center gap-2 text-sm font-medium text-slate-300 hover:text-white transition-colors focus:outline-none group"
                  >
                    <div className="relative w-8 h-8 flex items-center justify-center">
                        <Hexagon className="w-8 h-8 text-cyan-500 fill-cyan-500/20 stroke-[1.5]" />
                        <div className="absolute w-2 h-2 bg-cyan-400 rounded-full animate-pulse"></div>
                    </div>
                    <span className="hidden lg:inline">Account</span>
                    <ChevronDown className={`w-4 h-4 transition-transform duration-200 ${isProfileOpen ? 'rotate-180' : ''}`} />
                  </button>

                  {/* Backdrop to close on click outside */}
                  {isProfileOpen && (
                    <div 
                      className="fixed inset-0 z-40 cursor-default" 
                      onClick={() => setIsProfileOpen(false)} 
                    />
                  )}

                  {/* Dropdown Menu */}
                  <AnimatePresence>
                    {isProfileOpen && (
                      <motion.div
                        initial={{ opacity: 0, y: 10, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 10, scale: 0.95 }}
                        transition={{ duration: 0.1 }}
                        className="absolute right-0 mt-2 w-64 bg-slate-900 border border-slate-800 rounded-xl shadow-2xl py-2 z-50 overflow-hidden"
                      >
                        <div className="px-4 py-3 border-b border-slate-800 bg-slate-900/50">
                          <p className="text-xs text-slate-500 font-medium uppercase tracking-wider mb-1">Signed in as</p>
                          <p className="text-sm text-white font-medium truncate" title={user.email}>{user.email}</p>
                        </div>

                        <div className="py-2">
                          <Link 
                            to="/dashboard"
                            className="flex items-center gap-3 px-4 py-2.5 text-sm text-slate-300 hover:text-white hover:bg-slate-800 transition-colors"
                            onClick={() => setIsProfileOpen(false)}
                          >
                            <LayoutDashboard className="w-4 h-4 text-cyan-400" />
                            Dashboard
                          </Link>
                          
                          <Link 
                            to="/download"
                            className="flex items-center gap-3 px-4 py-2.5 text-sm text-slate-300 hover:text-white hover:bg-slate-800 transition-colors"
                            onClick={() => setIsProfileOpen(false)}
                          >
                            <Download className="w-4 h-4 text-purple-400" />
                            Downloads
                          </Link>

                          {user.is_admin && (
                            <Link 
                              to="/admin"
                              className="flex items-center gap-3 px-4 py-2.5 text-sm text-slate-300 hover:text-yellow-300 hover:bg-slate-800 transition-colors"
                              onClick={() => setIsProfileOpen(false)}
                            >
                              <Shield className="w-4 h-4 text-yellow-400" />
                              Admin Console
                            </Link>
                          )}
                        </div>

                        <div className="border-t border-slate-800 mt-1 pt-1">
                          <button 
                            onClick={handleLogout}
                            className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-red-400 hover:text-red-300 hover:bg-red-900/10 transition-colors text-left"
                          >
                            <LogOut className="w-4 h-4" />
                            Sign Out
                          </button>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </div>
            ) : (
              <>
                <Link
                  to="/login"
                  className="text-sm font-medium text-slate-300 hover:text-white transition-colors whitespace-nowrap"
                >
                  Log In
                </Link>
                <Link
                  to="/signup"
                  className="bg-cyan-500 hover:bg-cyan-400 text-white px-5 py-2 rounded-full text-sm font-semibold transition-all shadow-[0_0_15px_rgba(6,182,212,0.5)] hover:shadow-[0_0_25px_rgba(6,182,212,0.7)] whitespace-nowrap"
                >
                  Sign Up
                </Link>
              </>
            )}
          </div>
        </nav>

        {/* Mobile Menu Button */}
        <button
          className="md:hidden text-white"
          onClick={() => setIsOpen(!isOpen)}
        >
          {isOpen ? <X /> : <Menu />}
        </button>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: '100vh' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="md:hidden fixed inset-0 top-[72px] bg-slate-950 border-t border-slate-800 z-40 overflow-y-auto"
          >
            <nav className="flex flex-col p-6 gap-6 items-center text-center">
              {navLinks.map((link) => (
                <a
                  key={link.name}
                  href={link.href}
                  className="text-slate-300 hover:text-cyan-400 text-xl font-medium w-full py-2"
                  onClick={() => setIsOpen(false)}
                >
                  {link.name}
                </a>
              ))}
              
              <div className="w-full h-px bg-slate-800 my-2"></div>

              {user ? (
                <>
                  <div className="text-sm text-slate-400 mb-4">
                    Signed in as <span className="text-white font-medium">{user.email}</span>
                  </div>
                  
                  <Link 
                    to="/chat"
                    className="bg-cyan-500 hover:bg-cyan-400 text-white py-3 px-6 rounded-xl font-bold text-lg shadow-lg shadow-cyan-500/20 w-full flex items-center justify-center gap-2"
                    onClick={() => setIsOpen(false)}
                  >
                    <Zap className="w-5 h-5" />
                    Live Demo
                  </Link>

                  <Link
                    to="/dashboard"
                    className="text-slate-300 hover:text-white text-xl font-medium w-full py-2 flex items-center justify-center gap-2"
                    onClick={() => setIsOpen(false)}
                  >
                    <LayoutDashboard className="w-5 h-5" />
                    Dashboard
                  </Link>

                  <Link
                    to="/download"
                    className="text-slate-300 hover:text-white text-xl font-medium w-full py-2 flex items-center justify-center gap-2"
                    onClick={() => setIsOpen(false)}
                  >
                    <Download className="w-5 h-5" />
                    Downloads
                  </Link>

                  {user.is_admin && (
                    <Link
                      to="/admin"
                      className="text-yellow-400 hover:text-yellow-300 text-xl font-medium w-full py-2 flex items-center justify-center gap-2"
                      onClick={() => setIsOpen(false)}
                    >
                      <Shield className="w-5 h-5" />
                      Admin Dashboard
                    </Link>
                  )}
                  <button
                    onClick={() => {
                      handleLogout();
                      setIsOpen(false);
                    }}
                    className="text-red-400 hover:text-red-300 text-xl font-medium w-full py-2 flex items-center justify-center gap-2"
                  >
                    <LogOut className="w-5 h-5" />
                    Logout
                  </button>
                </>
              ) : (
                <div className="flex flex-col w-full gap-4">
                  <Link
                    to="/login"
                    className="text-slate-300 hover:text-white text-xl font-medium w-full py-2"
                    onClick={() => setIsOpen(false)}
                  >
                    Log In
                  </Link>
                  <Link
                    to="/signup"
                    className="bg-cyan-500 hover:bg-cyan-400 text-white py-4 rounded-xl font-bold text-lg shadow-lg shadow-cyan-500/20 w-full"
                    onClick={() => setIsOpen(false)}
                  >
                    Sign Up
                  </Link>
                </div>
              )}
            </nav>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
};
