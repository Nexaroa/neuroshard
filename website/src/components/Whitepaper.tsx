import { useState, useEffect } from 'react';
import { Download, FileText, Lock, Shield, BookOpen, Loader2 } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
const API_BASE = import.meta.env.VITE_API_URL || '';

export const Whitepaper = () => {
  const { user } = useAuth();
  const [pdfBlobUrl, setPdfBlobUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Protected PDF URL - requires authentication
  const pdfUrl = `${API_BASE}/api/whitepaper/pdf`;
  
  // Fetch PDF on mount with authentication
  useEffect(() => {
    const fetchPdf = async () => {
      try {
        setLoading(true);
        setError(null);
        const token = localStorage.getItem('token');
        
        if (!token) {
          setError('Authentication required');
          setLoading(false);
          return;
        }
        
        const response = await fetch(pdfUrl, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });
        
        if (!response.ok) {
          throw new Error('Failed to load whitepaper');
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        setPdfBlobUrl(url);
      } catch (err) {
        console.error('PDF fetch error:', err);
        setError('Failed to load whitepaper. Please try again.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchPdf();
    
    // Cleanup blob URL on unmount
    return () => {
      if (pdfBlobUrl) {
        window.URL.revokeObjectURL(pdfBlobUrl);
      }
    };
  }, [pdfUrl]);
  
  const handleDownload = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(pdfUrl, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (!response.ok) {
        throw new Error('Failed to download whitepaper');
      }
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'NeuroShard_Whitepaper.pdf';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Download error:', error);
      alert('Failed to download whitepaper. Please try again.');
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 pt-24 pb-12 px-4">
      <div className="container mx-auto max-w-5xl">
        {/* Header Section */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2 bg-cyan-500/20 rounded-lg">
                <BookOpen className="w-6 h-6 text-cyan-400" />
              </div>
              <h1 className="text-3xl font-bold text-white">Technical Whitepaper</h1>
            </div>
            <p className="text-slate-400">
              Complete technical documentation of the NeuroShard protocol
            </p>
          </div>
          
          <button
            onClick={handleDownload}
            className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white rounded-xl font-bold transition-all shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/40"
          >
            <Download className="w-5 h-5" />
            Download PDF
          </button>
        </div>

        {/* Access Badge */}
        <div className="mb-6 flex items-center gap-2 text-sm">
          <div className="flex items-center gap-2 px-3 py-1.5 bg-green-500/10 border border-green-500/30 rounded-full">
            <Shield className="w-4 h-4 text-green-400" />
            <span className="text-green-400">Authenticated Access</span>
          </div>
          <span className="text-slate-500">•</span>
          <span className="text-slate-400">Logged in as {user?.email}</span>
        </div>

        {/* PDF Viewer */}
        <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden shadow-2xl">
          <div className="bg-slate-800/50 px-4 py-3 border-b border-slate-700 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FileText className="w-4 h-4 text-slate-400" />
              <span className="text-slate-300 text-sm font-medium">NeuroShard_Whitepaper.pdf</span>
            </div>
            <span className="text-xs text-slate-500">Members Only</span>
          </div>
          <div className="h-[75vh]">
            {loading ? (
              <div className="w-full h-full flex items-center justify-center bg-slate-950">
                <div className="flex flex-col items-center gap-4">
                  <Loader2 className="w-10 h-10 text-cyan-400 animate-spin" />
                  <span className="text-slate-400">Loading whitepaper...</span>
                </div>
              </div>
            ) : error ? (
              <div className="w-full h-full flex items-center justify-center bg-slate-950">
                <div className="flex flex-col items-center gap-4 text-center px-4">
                  <div className="p-4 bg-red-500/10 rounded-full">
                    <Lock className="w-10 h-10 text-red-400" />
                  </div>
                  <span className="text-red-400 font-medium">{error}</span>
                  <button
                    onClick={() => window.location.reload()}
                    className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white rounded-lg text-sm transition-colors"
                  >
                    Try Again
                  </button>
                </div>
              </div>
            ) : pdfBlobUrl ? (
              <iframe
                src={`${pdfBlobUrl}#toolbar=1&navpanes=0`}
                className="w-full h-full"
                title="NeuroShard Whitepaper"
              />
            ) : null}
          </div>
        </div>

        {/* Footer Note */}
        <div className="mt-6 text-center text-slate-500 text-sm">
          <p>
            This document is confidential and intended for registered NeuroShard members only.
            <br />
            Please do not share without permission.
          </p>
        </div>
      </div>
    </div>
  );
};
