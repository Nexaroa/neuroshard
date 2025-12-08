import { Download, FileText, BookOpen } from 'lucide-react';

export const Whitepaper = () => {
  // Public PDF URL - directly from GitHub
  const pdfUrl = 'https://github.com/Nexaroa/neuroshard/raw/main/docs/whitepaper/neuroshard_whitepaper.pdf';
  
  const handleDownload = () => {
    const a = document.createElement('a');
    a.href = pdfUrl;
    a.download = 'NeuroShard_Whitepaper.pdf';
    a.target = '_blank';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
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

        {/* Public Access Badge */}
        <div className="mb-6 flex items-center gap-2 text-sm">
          <div className="flex items-center gap-2 px-3 py-1.5 bg-cyan-500/10 border border-cyan-500/30 rounded-full">
            <FileText className="w-4 h-4 text-cyan-400" />
            <span className="text-cyan-400">Public Document</span>
          </div>
        </div>

        {/* PDF Viewer */}
        <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden shadow-2xl">
          <div className="bg-slate-800/50 px-4 py-3 border-b border-slate-700 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FileText className="w-4 h-4 text-slate-400" />
              <span className="text-slate-300 text-sm font-medium">NeuroShard_Whitepaper.pdf</span>
            </div>
            <span className="text-xs text-slate-500">Open Source</span>
          </div>
          <div className="h-[75vh]">
            <iframe
              src={`${pdfUrl}#toolbar=1&navpanes=0`}
              className="w-full h-full"
              title="NeuroShard Whitepaper"
            />
          </div>
        </div>

        {/* Footer Note */}
        <div className="mt-6 text-center text-slate-500 text-sm">
          <p>
            This whitepaper is open source and publicly available.
            <br />
            <a 
              href="https://github.com/Nexaroa/neuroshard" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-cyan-400 hover:text-cyan-300 underline"
            >
              View source code on GitHub
            </a>
          </p>
        </div>
      </div>
    </div>
  );
};
