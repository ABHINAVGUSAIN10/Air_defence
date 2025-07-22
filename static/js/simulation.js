document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('radarCanvas');
    const ctx = canvas.getContext('2d');
    const infoPanel = document.getElementById('info-panel');

    const RADAR_CENTER = { x: canvas.width / 2, y: canvas.height / 2 };
    const RADAR_RADIUS = canvas.width / 2 - 50;
    const BASE_POS = { x: RADAR_CENTER.x, y: RADAR_CENTER.y };

    let planes = [];
    let missiles = [];
    let explosions = [];

    function drawRadarGrid() {
        ctx.strokeStyle = 'rgba(0, 255, 0, 0.3)';
        ctx.lineWidth = 1;
        for (let i = 1; i <= 4; i++) {
            ctx.beginPath();
            ctx.arc(RADAR_CENTER.x, RADAR_CENTER.y, (RADAR_RADIUS / 4) * i, 0, Math.PI * 2);
            ctx.stroke();
        }
        for (let i = 0; i < 12; i++) {
            const angle = (i * 30 * Math.PI) / 180;
            ctx.beginPath();
            ctx.moveTo(RADAR_CENTER.x, RADAR_CENTER.y);
            ctx.lineTo(RADAR_CENTER.x + RADAR_RADIUS * Math.cos(angle), RADAR_CENTER.y + RADAR_RADIUS * Math.sin(angle));
            ctx.stroke();
        }
    }

    class Plane {
        constructor(id, name, modulation, initialState, isThreat) {
            this.id = id;
            this.name = name;
            this.modulation = modulation;
            this.isThreat = isThreat;
            this.x = initialState.x;
            this.y = initialState.y;
            this.vx = initialState.vx;
            this.vy = initialState.vy;
            this.status = "Active";
            this.missileFired = false;
        }

        update() {
            if (this.status === "Active") {
                // Update position based on velocity
                this.x += this.vx;
                this.y += this.vy;
            }
            const distFromCenter = Math.hypot(this.x - RADAR_CENTER.x, this.y - RADAR_CENTER.y);
            if (this.isThreat && this.status === "Active" && !this.missileFired && distFromCenter < RADAR_RADIUS) {
                missiles.push(new Missile(BASE_POS.x, BASE_POS.y, this));
                this.missileFired = true;
            }
        }

        draw() {
            if (this.status !== "Active") return;
            ctx.fillStyle = this.isThreat ? 'red' : 'yellow';
            ctx.beginPath();
            ctx.moveTo(this.x, this.y - 8);
            ctx.lineTo(this.x - 6, this.y + 6);
            ctx.lineTo(this.x + 6, this.y + 6);
            ctx.closePath();
            ctx.fill();
        }
    }

    class Missile {
        constructor(x, y, target) {
            this.x = x; this.y = y; this.target = target;
            this.speed = 10; this.active = true;
        }

        update() {
            if (!this.active || this.target.status !== "Active") {
                this.active = false; return;
            }
            const angle = Math.atan2(this.target.y - this.y, this.target.x - this.x);
            this.x += Math.cos(angle) * this.speed;
            this.y += Math.sin(angle) * this.speed;

            if (Math.hypot(this.x - this.target.x, this.y - this.target.y) < 10) {
                this.active = false;
                this.target.status = "Destroyed";
                explosions.push(new Explosion(this.target.x, this.target.y));
            }
        }

        draw() {
            if (!this.active) return;
            ctx.fillStyle = 'white'; ctx.beginPath();
            ctx.arc(this.x, this.y, 2, 0, Math.PI * 2); ctx.fill();
        }
    }

    class Explosion {
        constructor(x, y) {
            this.x = x; this.y = y; this.radius = 5;
            this.alpha = 1.0;
        }
        update() {
            this.radius += 0.5; this.alpha -= 0.02;
        }
        draw() {
            if (this.alpha <= 0) return;
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 165, 0, ${this.alpha})`;
            ctx.fill();
        }
    }

    function createInfoPanel() {
        infoPanel.innerHTML = '<h2>Aircraft Status</h2>';
        planes.forEach(plane => {
            const card = document.createElement('div');
            card.className = 'info-card' + (plane.isThreat ? ' threat' : '');
            card.id = `info-${plane.id}`;
            card.innerHTML = `
                <h4>${plane.name}</h4>
                <p><strong>Status:</strong> <span class="status">${plane.status}</span></p>
                <p><strong>Modulation:</strong> ${plane.modulation}</p>
                <p><strong>Position:</strong> (<span class="pos-x">${plane.x.toFixed(0)}</span>, <span class="pos-y">${plane.y.toFixed(0)}</span>)</p>
            `;
            infoPanel.appendChild(card);
        });
    }

    function updateInfoPanel() {
        planes.forEach(plane => {
            const card = document.getElementById(`info-${plane.id}`);
            if (card) {
                card.querySelector('.status').textContent = plane.status;
                card.querySelector('.pos-x').textContent = plane.x.toFixed(0);
                card.querySelector('.pos-y').textContent = plane.y.toFixed(0);
                if (plane.status === "Destroyed") {
                    card.classList.add('destroyed');
                }
            }
        });
    }

    function init() {
        if (typeof simulationData !== 'undefined' && Array.isArray(simulationData)) {
            simulationData.forEach(p => {
                planes.push(new Plane(p.id, p.name, p.modulation, p.initial_state, p.is_threat));
            });
            createInfoPanel();
        } else {
            console.error("Simulation data is missing or invalid.");
            return;
        }
        gameLoop();
    }
    
    let sweepAngle = 0;
    function gameLoop() {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        drawRadarGrid();

        ctx.fillStyle = '#006400';
        ctx.fillRect(BASE_POS.x - 15, BASE_POS.y - 10, 30, 20);
        ctx.fillStyle = 'white';
        ctx.font = '12px sans-serif';
        ctx.fillText('BASE', BASE_POS.x - 12, BASE_POS.y + 4);

        planes.forEach(p => { p.update(); p.draw(); });
        missiles.forEach(m => { m.update(); m.draw(); });
        explosions.forEach(e => { e.update(); e.draw(); });
        
        missiles = missiles.filter(m => m.active);
        explosions = explosions.filter(e => e.alpha > 0);
        
        updateInfoPanel();

        sweepAngle = (sweepAngle + 0.015) % (Math.PI * 2);
        const gradient = ctx.createRadialGradient(RADAR_CENTER.x, RADAR_CENTER.y, 10, RADAR_CENTER.x, RADAR_CENTER.y, RADAR_RADIUS);
        gradient.addColorStop(0, 'rgba(0, 255, 0, 0.4)');
        gradient.addColorStop(0.9, 'rgba(0, 255, 0, 0.1)');
        gradient.addColorStop(1, 'rgba(0, 255, 0, 0)');
        ctx.beginPath();
        ctx.moveTo(RADAR_CENTER.x, RADAR_CENTER.y);
        ctx.arc(RADAR_CENTER.x, RADAR_CENTER.y, RADAR_RADIUS, sweepAngle, sweepAngle + Math.PI / 2);
        ctx.closePath();
        ctx.fillStyle = gradient;
        ctx.fill();

        requestAnimationFrame(gameLoop);
    }
    
    init();
});