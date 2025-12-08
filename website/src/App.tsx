import { Routes, Route } from 'react-router-dom';
import { Header } from './components/Header';
import { Hero } from './components/Hero';
import { Vision } from './components/Vision';
// import { Features } from './components/Features';
import { HowItWorks } from './components/HowItWorks';
import { Technology } from './components/Technology';
import { Download } from './components/Download';
import { Footer } from './components/Footer';
import { Login } from './components/Login';
import { Signup } from './components/Signup';
import { DownloadPage } from './components/DownloadPage';
import { Whitepaper } from './components/Whitepaper';
import { ProtectedRoute } from './components/ProtectedRoute';
import { AdminDashboard } from './components/AdminDashboard';
import { Chat } from './components/Chat';
import { UserDashboard } from './components/UserDashboard';
import { LedgerExplorer } from './components/LedgerExplorer';
import { Legal } from './components/Legal';
import { NotFound } from './components/NotFound';
import { ScrollToTop } from './components/ScrollToTop';
import { GlobalLLMStatus } from './components/GlobalLLMStatus';
import { WaitlistSignup } from './components/WaitlistSignup';
import { SEO } from './components/SEO';

function Home() {
  return (
    <main>
      <SEO />
      <Hero />
      <Vision />
      {/* <Features /> */}
      <HowItWorks />
      <Technology />
      <Download />
    </main>
  );
}

function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-white selection:bg-cyan-500/30">
      <ScrollToTop />
      <Header />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/join" element={<WaitlistSignup />} />
        <Route
          path="/chat"
          element={
            <ProtectedRoute>
              <Chat />
            </ProtectedRoute>
          }
        />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/legal" element={<Legal />} />
        <Route path="/whitepaper" element={<Whitepaper />} />
        <Route path="/ledger" element={<LedgerExplorer />} />
        <Route path="/training" element={<GlobalLLMStatus />} />
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <UserDashboard />
            </ProtectedRoute>
          }
        />
        <Route
          path="/download"
          element={
            <ProtectedRoute>
              <DownloadPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/admin"
          element={
            <ProtectedRoute>
              <AdminDashboard />
            </ProtectedRoute>
          }
        />
        {/* 404 Catch-all */}
        <Route path="*" element={<NotFound />} />
      </Routes>
      <Footer />
    </div>
  );
}

export default App;
