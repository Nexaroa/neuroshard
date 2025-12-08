import { useEffect, useRef } from 'react';

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  alpha: number;
  connected: boolean;
}

export const NetworkBackground = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let particles: Particle[] = [];
    const GRID_SIZE = 60;

    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    const createParticle = (y?: number): Particle => {
      // Snap to vertical grid lines
      // Calculate total columns
      const cols = Math.ceil(canvas.width / GRID_SIZE);
      // Pick a random column
      const col = Math.floor(Math.random() * cols);
      // Calculate x position centered on the grid line
      // Use modulo to center the grid on screen
      const xOffset = (canvas.width % GRID_SIZE) / 2;
      const x = col * GRID_SIZE + xOffset;

      return {
        x: x,
        y: y ?? canvas.height + Math.random() * 100,
        vx: 0, // No horizontal drift - stay on the line
        vy: -0.5 - Math.random() * 1.5, // Upward movement
        size: Math.random() * 2 + 1.5,
        alpha: Math.random() * 0.5 + 0.2,
        connected: false
      };
    };

    const initParticles = () => {
      particles = [];
      const particleCount = Math.min(Math.floor(window.innerWidth * 0.15), 120);
      for (let i = 0; i < particleCount; i++) {
        particles.push(createParticle(Math.random() * canvas.height));
      }
    };

    const drawGrid = () => {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.07)';
        ctx.lineWidth = 1;
        
        const xOffset = (canvas.width % GRID_SIZE) / 2;
        
        // Vertical lines
        for (let x = xOffset; x <= canvas.width; x += GRID_SIZE) {
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, canvas.height);
          ctx.stroke();
        }
        
        // Horizontal lines
        for (let y = 0; y <= canvas.height; y += GRID_SIZE) {
          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(canvas.width, y);
          ctx.stroke();
        }
    }

    const drawParticles = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      drawGrid();

      // Update and draw particles
      particles.forEach((p, i) => {
        p.y += p.vy;

        // Respawn at bottom if goes off top
        if (p.y < -50) {
          Object.assign(p, createParticle());
        }

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(6, 182, 212, ${p.alpha})`; // Cyan color
        ctx.fill();
        
        // Add a glow effect to the particle itself
        ctx.shadowBlur = 15;
        ctx.shadowColor = "rgba(6, 182, 212, 0.5)";
        ctx.fill();
        ctx.shadowBlur = 0;

        // Connections
        p.connected = false;
        for (let j = i + 1; j < particles.length; j++) {
          const p2 = particles[j];
          const dx = p.x - p2.x;
          const dy = p.y - p2.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          // Connect if close enough
          // Since they are on grid lines, most connections will be vertical or diagonal neighbors
          if (distance < GRID_SIZE * 2.5) {
            p.connected = true;
            p2.connected = true;
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p2.x, p2.y);
            const opacity = 1 - distance / (GRID_SIZE * 2.5);
            ctx.strokeStyle = `rgba(6, 182, 212, ${opacity * 0.4})`;
            ctx.lineWidth = 1;
            ctx.stroke();
            
            // "Pulse" effect on connection
             if (distance < GRID_SIZE) {
                ctx.shadowBlur = 8;
                ctx.shadowColor = "rgba(6, 182, 212, 0.6)";
                ctx.stroke();
                ctx.shadowBlur = 0;
             }
          }
        }
      });

      animationFrameId = requestAnimationFrame(drawParticles);
    };

    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    initParticles();
    drawParticles();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none [mask-image:linear-gradient(180deg,white,rgba(255,255,255,0))]"
    />
  );
};
