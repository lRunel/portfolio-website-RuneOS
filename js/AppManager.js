// AppManager instances maps name to { win, pid }
const AppManager = {
    instances: {},
    _winCount: 0,

    launch: (appName, args = null) => {
        if (!System.booted) return;

        // Singleton Logic for main apps
        if (['terminal', 'projects', 'resume', 'settings', 'mail', 'overview'].includes(appName) && AppManager.instances[appName]) {
            if (WindowManager.restoreMinimized(appName)) return;
            AppManager.focus(AppManager.instances[appName].win);
            return;
        }

        let pid;
        try {
            pid = Shell.spawn(appName);
        } catch (e) {
            console.error('[AppManager] Shell.spawn failed:', e);
            return;
        }
        if (!pid || pid < 0) {
            console.error('[AppManager] Invalid PID from spawn:', pid);
            return;
        }

        const win = document.createElement('article');
        win.className = 'window active';
        win.setAttribute('role', 'dialog');
        win.dataset.pid = pid;
        win.style.zIndex = ++System.zIndex;

        // Per-app default dimensions & cascaded stacking
        const SIZES = {
            'overview': { w: 680, h: 620 },
            'terminal': { w: 720, h: 460 },
            'projects': { w: 940, h: 620 },
            'resume':   { w: 900, h: 700 },
            'settings': { w: 520, h: 420 },
            'mail':     { w: 480, h: 470 }
        };

        const isMobile = window.innerWidth <= 768;
        if (!isMobile) {
            AppManager._winCount = (AppManager._winCount || 0) + 1;
            const offset = (AppManager._winCount * 28) % 140;
            const dim = SIZES[appName] || { w: 680, h: 450 };
            const targetW = Math.min(dim.w, window.innerWidth - 40);
            const targetH = Math.min(dim.h, window.innerHeight - 120);
            win.style.width = targetW + 'px';
            win.style.height = targetH + 'px';
            const baseLeft = Math.max(10, Math.round((window.innerWidth - targetW) / 2));
            const baseTop = Math.max(20, Math.round((window.innerHeight - targetH - 60) / 2));
            win.style.left = Math.min(baseLeft + offset, window.innerWidth - targetW - 10) + 'px';
            win.style.top = Math.min(baseTop + offset, window.innerHeight - targetH - 90) + 'px';
        }

        // Title Mapping
        const titles = {
            'overview': 'About · Shubham R Chowta',
            'terminal': 'Terminal',
            'projects': 'Projects',
            'resume': 'Resume',
            'settings': 'Settings',
            'mail': 'Contact Me',
            'text-viewer': args?.filename || 'Text Viewer',
            'html-viewer': args?.title || 'Document Viewer'
        };

        // Window HTML
        win.innerHTML = `
            <div class="win-header">
                <span class="material-symbols-rounded back-btn">arrow_back</span>
                <span class="win-title">${titles[appName] || 'Window'}</span>
                <div class="win-controls">
                    <button class="win-btn minimize" aria-label="Minimize"><span class="material-symbols-rounded" style="font-size:16px">remove</span></button>
                    <button class="win-btn maximize" aria-label="Maximize"><span class="material-symbols-rounded" style="font-size:14px">check_box_outline_blank</span></button>
                    <button class="win-btn close" aria-label="Close"><span class="material-symbols-rounded" style="font-size:16px">close</span></button>
                </div>
            </div>
            <div class="win-content"></div>
        `;

        // Load Content
        const contentArea = win.querySelector('.win-content');
        if (appName === 'overview') AppManager.renderOverview(contentArea, pid);
        else if (appName === 'terminal') AppManager.renderTerminal(contentArea, pid);
        else if (appName === 'projects') AppManager.renderProjects(contentArea, pid);
        else if (appName === 'resume') AppManager.renderResume(contentArea, pid);
        else if (appName === 'settings') AppManager.renderSettings(contentArea, pid);
        else if (appName === 'mail') AppManager.renderMail(contentArea, pid);
        else if (appName === 'text-viewer') AppManager.renderTextViewer(contentArea, args, pid);
        else if (appName === 'html-viewer') AppManager.renderHtmlViewer(contentArea, args, pid);

        // Inject
        const container = isMobile ? document.getElementById('mobile-env') : document.getElementById('workspace');
        container.appendChild(win);

        // Add event listeners programmatically (Pointer Events for dragging)
        const headerEl = win.querySelector('.win-header');
        headerEl.addEventListener('pointerdown', (e) => WindowManager.startDrag(e, headerEl));
        win.querySelector('.back-btn').addEventListener('click', () => AppManager.close(win));
        win.querySelector('.minimize').addEventListener('click', (e) => WindowManager.toggleMinimize(e.currentTarget));
        win.querySelector('.maximize').addEventListener('click', (e) => WindowManager.toggleMaximize(e.currentTarget));
        win.querySelector('.close').addEventListener('click', () => AppManager.close(win));

        // Register (unless temporary util like text viewer)
        if (!args?.temp) {
            AppManager.instances[appName] = { win, pid };
            // Update Dock
            const dockItem = document.getElementById(`dock-${appName}`);
            if (dockItem) dockItem.classList.add('running');
        }

        // Launch minimized if requested
        if (args?.minimized) {
            win.classList.add('minimized');
            win.classList.remove('active');
        }

        // Focus Listener
        win.onmousedown = () => AppManager.focus(win);
    },

    /* Mounts hairline line-figures into every [data-figure] element under root.
       Figures are stored on their window so close() can stop their animation loops. */
    mountFigures: (root) => {
        const nodes = root.querySelectorAll('[data-figure]');
        if (!nodes.length) return;
        const url = new URL('js/vendor/hairline.js', document.baseURI).href;
        import(url).then(hairline => {
            const win = root.closest('.window');
            nodes.forEach(el => {
                const make = hairline[el.dataset.figure];
                if (!make || el._figure) return;
                el._figure = make(el, { theme: 'dark', intensity: 0.7, play: el.dataset.play === 'true', label: el.dataset.label || undefined });
                if (win) (win._figures = win._figures || []).push(el._figure);
            });
        }).catch(err => console.warn('[hairline] failed to load:', err));
    },

    close: (btnOrName) => {
        let targetWin;

        if (typeof btnOrName === 'string') {
            targetWin = AppManager.instances[btnOrName]?.win;
        } else if (btnOrName instanceof HTMLElement) {
            targetWin = btnOrName.closest ? btnOrName.closest('.window') : btnOrName;
        } else {
            return;
        }

        if (!targetWin) return;

        const pid = parseInt(targetWin.dataset.pid);

        // Immediately unregister so app can be relaunched
        for (const [key, val] of Object.entries(AppManager.instances)) {
            if (val.win === targetWin) {
                delete AppManager.instances[key];
                const dockItem = document.getElementById(`dock-${key}`);
                if (dockItem) dockItem.classList.remove('running');
            }
        }

        // Kill process AFTER unregistering to avoid race conditions
        if (!isNaN(pid) && pid > 0) {
            try { Shell.exit(pid); } catch (e) { /* process may already be dead */ }
        }

        (targetWin._figures || []).forEach(f => { try { f.destroy(); } catch (e) {} });
        targetWin._figures = [];

        // Smooth close animation via CSS transition
        targetWin.style.transform = 'scale(0.92)';
        targetWin.style.opacity = '0';
        targetWin.style.pointerEvents = 'none';

        setTimeout(() => {
            if (targetWin.parentNode) targetWin.parentNode.removeChild(targetWin);
        }, 200);
    },

    focus: (win) => {
        document.querySelectorAll('.window').forEach(w => w.classList.remove('active'));
        win.classList.add('active');
        win.style.zIndex = ++System.zIndex;
    },

    /* APP RENDERERS */
    renderOverview: (container, pid) => {
        container.innerHTML = `
            <div class="overview-container">
                <header class="overview-header">
                    <div class="overview-profile">
                        <div class="overview-avatar">SC</div>
                        <div class="overview-id">
                            <h2 class="overview-name">Shubham R Chowta</h2>
                            <p class="overview-role">Backend & Systems Engineer · Python, C, AWS, ML</p>
                            <p class="overview-location"><span class="material-symbols-rounded" style="font-size:15px;vertical-align:text-bottom">location_on</span> Mangaluru / Udupi, India · Open to Relocation & Remote</p>
                        </div>
                    </div>
                </header>

                <section class="overview-section">
                    <div class="overview-label">Education</div>
                    <div class="overview-card-row">
                        <div class="overview-school">Sahyadri College of Engineering & Management <span class="overview-years">(2023–2027)</span></div>
                        <div class="overview-gpa"><span class="gpa-badge">CGPA: 8.77 / 10.0</span> · B.Tech Computer Science (AI & ML)</div>
                    </div>
                </section>

                <section class="overview-section">
                    <div class="overview-label">Selected Results</div>
                    <div class="overview-metrics-grid">
                        <div class="metric-card">
                            <div class="metric-val">+9.45 pts</div>
                            <div class="metric-desc">SA2: 94.07% vs 84.62% MNIST accuracy over synchronous training, 6 laptops &amp; phones, 5-min run</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-val">98.5%</div>
                            <div class="metric-desc">Arecanut detector: validation accuracy, 9 classes, 4,000+ images; ~70% smaller after INT8 quantization</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-val">131</div>
                            <div class="metric-desc">AWSolotl: pytest tests (245 assertions) covering the policy checks and audit log</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-val">1 of 60</div>
                            <div class="metric-desc">Selected from 600 applicants for the SE Training Program; Scrum Master of a 5-person team</div>
                        </div>
                    </div>
                </section>

                <section class="overview-section">
                    <div class="overview-label">Quick Links</div>
                    <div class="overview-actions-grid">
                        <a href="assets/Shubham_Chowta_SWE_Backend.pdf" download="Shubham_Chowta_SWE_Backend.pdf" class="overview-btn primary">
                            <span class="material-symbols-rounded">download</span> Download Resume (PDF)
                        </a>
                        <button class="overview-btn" id="ov-btn-projects">
                            <span class="material-symbols-rounded">folder_open</span> Browse Projects
                        </button>
                        <button class="overview-btn" id="ov-btn-copy-email" data-email="shubhamchouta@gmail.com">
                            <span class="material-symbols-rounded">content_copy</span> <span id="ov-email-text">Copy Email</span>
                        </button>
                        <a href="https://www.linkedin.com/in/shubham-chowta/" target="_blank" rel="noopener" class="overview-btn">
                            <span class="material-symbols-rounded">link</span> LinkedIn
                        </a>
                        <a href="https://github.com/lRunel" target="_blank" rel="noopener" class="overview-btn">
                            <span class="material-symbols-rounded">code</span> GitHub
                        </a>
                        <button class="overview-btn" id="ov-btn-terminal">
                            <span class="material-symbols-rounded">terminal</span> Open Terminal
                        </button>
                    </div>
                </section>

                <section class="overview-section">
                    <div class="overview-label">Technical Stack</div>
                    <div class="overview-stack-categories">
                        <div class="stack-row">
                            <span class="stack-category">Languages:</span>
                            <div class="stack-chips">
                                <span class="chip">Python</span>
                                <span class="chip">C/C++</span>
                                <span class="chip">SQL</span>
                                <span class="chip">Java</span>
                                <span class="chip">JavaScript</span>
                            </div>
                        </div>
                        <div class="stack-row">
                            <span class="stack-category">Backend & Cloud:</span>
                            <div class="stack-chips">
                                <span class="chip">FastAPI</span>
                                <span class="chip">AWS (DynamoDB, S3, IAM, ECS, CDK)</span>
                                <span class="chip">Docker</span>
                                <span class="chip">Linux</span>
                            </div>
                        </div>
                        <div class="stack-row">
                            <span class="stack-category">ML & Data:</span>
                            <div class="stack-chips">
                                <span class="chip">PyTorch</span>
                                <span class="chip">TensorFlow / TFLite</span>
                                <span class="chip">Reinforcement Learning</span>
                                <span class="chip">Pandas / Polars</span>
                            </div>
                        </div>
                    </div>
                </section>
            </div>
        `;

        const projBtn = container.querySelector('#ov-btn-projects');
        if (projBtn) projBtn.addEventListener('click', () => AppManager.launch('projects'));

        const termBtn = container.querySelector('#ov-btn-terminal');
        if (termBtn) termBtn.addEventListener('click', () => AppManager.launch('terminal'));

        const copyBtn = container.querySelector('#ov-btn-copy-email');
        if (copyBtn) {
            copyBtn.addEventListener('click', () => {
                const email = copyBtn.dataset.email;
                navigator.clipboard.writeText(email).then(() => {
                    const textSpan = container.querySelector('#ov-email-text');
                    if (textSpan) {
                        const original = textSpan.textContent;
                        textSpan.textContent = 'Copied!';
                        setTimeout(() => { textSpan.textContent = original; }, 2000);
                    }
                }).catch(() => {
                    window.location.href = 'mailto:' + email;
                });
            });
        }
    },

    renderTerminal: (container, pid) => {
        const initCwd = Shell.getcwd(pid);
        container.innerHTML = `
            <div class="term-container" onclick="if (!window.getSelection().toString()) this.querySelector('input').focus()">
                <div class="term-output"></div>
                <div class="cat-prompt" style="color: #ffb86c; margin-top: 4px; display: none;">=^.^=</div>
                <div class="cmd-line" style="display: none;">
                    <span class="prompt" style="color: #ffb86c">rune:${initCwd.replace('/home', '~')}$</span>
                    <input class="term-input" type="text" autocomplete="off" spellcheck="false">
                </div>
            </div>
        `;
        const input = container.querySelector('input');
        const output = container.querySelector('.term-output');
        const termContainer = container.querySelector('.term-container');

        /* ── Helper: append styled AI line to terminal ── */
        const appendAI = (text, cssClass) => {
            const line = document.createElement('div');
            line.className = 'ai-msg ' + (cssClass || '');
            line.innerHTML = '<span class="ai-prefix">[AI]</span> <span class="ai-body">' + text + '</span>';
            output.appendChild(line);
            termContainer.scrollTop = termContainer.scrollHeight;
        };

        const showIntro = () => {
            const intro = ` /\\_/\\
( o.o )
 > ^ <

Rune-os — a C kernel compiled to WebAssembly
Shubham R Chowta · Backend & Systems Engineer
Commands run on the kernel. Try 'ls', 'ps', 'cat /kernel/state', or 'help':`;

            output.innerHTML = `
                <div class="term-intro-text">${intro.replace(/\n/g, '<br>')}</div>
                <div class="terminal-quick-pills">
                    <button class="term-pill" data-action="projects" aria-label="Projects"><span class="pill-icon">📂</span> Projects</button>
                    <button class="term-pill" data-action="resume" aria-label="Resume"><span class="pill-icon">📄</span> Resume</button>
                    <button class="term-pill" data-action="cache" aria-label="Ask Cache"><span class="pill-icon">🐱</span> Ask Cache (scripted)</button>
                    <button class="term-pill" data-action="contact" aria-label="Contact"><span class="pill-icon">✉️</span> Contact</button>
                    <button class="term-pill" data-action="download" aria-label="Download PDF"><span class="pill-icon">💾</span> Download PDF</button>
                </div>
            `;
            container.querySelector('.cat-prompt').style.display = 'block';
            container.querySelector('.cmd-line').style.display = 'flex';
            input.focus();
            termContainer.scrollTop = termContainer.scrollHeight;
        };

        /* ── Click listener for interactive pills ── */
        output.addEventListener('click', (e) => {
            const pill = e.target.closest('.term-pill');
            if (!pill) return;
            const action = pill.dataset.action;
            if (action === 'projects') {
                AppManager.launch('projects');
                output.innerHTML += '<div class="term-system-line">[System] Opened Projects window.</div>';
            } else if (action === 'resume') {
                AppManager.launch('resume');
                output.innerHTML += '<div class="term-system-line">[System] Opened Resume window.</div>';
            } else if (action === 'cache') {
                input.value = 'cache ';
                input.focus();
            } else if (action === 'contact') {
                AppManager.launch('mail');
                output.innerHTML += '<div class="term-system-line">[System] Opened Contact dialog (shubhamchouta@gmail.com).</div>';
            } else if (action === 'download') {
                const a = document.createElement('a');
                a.href = 'assets/Shubham_Chowta_SWE_Backend.pdf';
                a.download = 'Shubham_Chowta_SWE_Backend.pdf';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                output.innerHTML += '<div class="term-system-line">[System] Downloading Shubham_Chowta_SWE_Backend.pdf...</div>';
            }
            termContainer.scrollTop = termContainer.scrollHeight;
        });

        /* ── Terminal Intro (Non-blocking) ── */
        showIntro();

        /* ── Command history & Tab autocomplete buffers ── */
        const cmdHistory = [];
        let historyIndex = -1;
        let historyDraft = '';

        /* ── Terminal input handler ── */
        input.onkeydown = (e) => {
            if (e.key === 'ArrowUp') {
                e.preventDefault();
                if (cmdHistory.length === 0) return;
                if (historyIndex === -1) historyDraft = input.value;
                if (historyIndex < cmdHistory.length - 1) {
                    historyIndex++;
                    input.value = cmdHistory[cmdHistory.length - 1 - historyIndex];
                }
                return;
            }

            if (e.key === 'ArrowDown') {
                e.preventDefault();
                if (historyIndex > 0) {
                    historyIndex--;
                    input.value = cmdHistory[cmdHistory.length - 1 - historyIndex];
                } else if (historyIndex === 0) {
                    historyIndex = -1;
                    input.value = historyDraft;
                }
                return;
            }

            if (e.key === 'Tab') {
                e.preventDefault();
                const text = input.value;
                const tokens = text.split(/\s+/);
                const lastToken = tokens[tokens.length - 1] || '';
                if (!lastToken) return;

                const candidates = [
                    'help', 'clear', 'reboot', 'projects', 'resume', 'contact', 'mail',
                    'overview', 'download', 'settings', 'cache', 'ps', 'ls', 'cd', 'cat',
                    'pwd', 'mkdir', 'touch', 'echo'
                ];
                try {
                    const lsOut = Shell.ls(pid, '.').data;
                    if (lsOut) {
                        lsOut.split('\n').map(s => s.trim()).filter(Boolean).forEach(f => {
                            if (!candidates.includes(f)) candidates.push(f);
                        });
                    }
                } catch (err) {}

                const matches = candidates.filter(c => c.startsWith(lastToken));
                if (matches.length === 1) {
                    tokens[tokens.length - 1] = matches[0];
                    input.value = tokens.join(' ') + ' ';
                } else if (matches.length > 1) {
                    const matchLine = document.createElement('div');
                    matchLine.style.cssText = 'color:#888;font-size:12px;margin:4px 0;font-family:monospace;';
                    matchLine.textContent = matches.join('  ');
                    output.appendChild(matchLine);
                    termContainer.scrollTop = termContainer.scrollHeight;
                }
                return;
            }

            if (e.key === 'Enter') {
                const val = input.value.trim();
                if (!val) return;

                if (cmdHistory.length === 0 || cmdHistory[cmdHistory.length - 1] !== val) {
                    cmdHistory.push(val);
                }
                historyIndex = -1;
                historyDraft = '';

                const currentCwd = Shell.getcwd(pid);
                const displayCwd = currentCwd.replace('/home', '~');
                output.innerText += '\n=^.^=\nrune:' + displayCwd + '$ ' + val;

                // JS-only commands
                if (val === 'clear') {
                    output.innerHTML = '';
                    input.value = '';
                    return;
                }
                if (val === 'reboot') {
                    sessionStorage.removeItem('rune_booted');
                    location.reload();
                    return;
                }
                if (val === 'overview') {
                    AppManager.launch('overview');
                    output.innerText += '\n[System] Opened About window.';
                    input.value = '';
                    termContainer.scrollTop = termContainer.scrollHeight;
                    return;
                }
                if (val === 'projects') {
                    AppManager.launch('projects');
                    output.innerText += '\n[System] Launched Projects window.';
                    input.value = '';
                    termContainer.scrollTop = termContainer.scrollHeight;
                    return;
                }
                if (val === 'resume') {
                    AppManager.launch('resume');
                    output.innerText += '\n[System] Launched Resume window.';
                    input.value = '';
                    termContainer.scrollTop = termContainer.scrollHeight;
                    return;
                }
                if (val === 'contact' || val === 'mail') {
                    AppManager.launch('mail');
                    output.innerText += '\n[System] Email: shubhamchouta@gmail.com | Phone: +91 79750 06501';
                    input.value = '';
                    termContainer.scrollTop = termContainer.scrollHeight;
                    return;
                }
                if (val === 'download') {
                    const a = document.createElement('a');
                    a.href = 'assets/Shubham_Chowta_SWE_Backend.pdf';
                    a.download = 'Shubham_Chowta_SWE_Backend.pdf';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    output.innerText += '\n[System] Downloading Shubham_Chowta_SWE_Backend.pdf...';
                    input.value = '';
                    termContainer.scrollTop = termContainer.scrollHeight;
                    return;
                }

                // All other commands -> real C kernel via WASM
                const response = Shell.exec(pid, val);

                /* Handle AI queries asynchronously */
                if (response.data === 'AI_QUERY' && response._prompt) {
                    const thinkLine = document.createElement('div');
                    thinkLine.className = 'ai-msg ai-response';
                    thinkLine.innerHTML = '<span class="ai-prefix">=^.^=</span> <span class="ai-body ai-thinking-text">thinking...</span>';
                    output.appendChild(thinkLine);
                    input.value = '';
                    input.disabled = true;
                    termContainer.scrollTop = termContainer.scrollHeight;

                    let firstTokenReceived = false;

                    const tokenListener = (data) => {
                        if (!firstTokenReceived) {
                            firstTokenReceived = true;
                            thinkLine.querySelector('.ai-body').innerText = '';
                            thinkLine.querySelector('.ai-body').classList.remove('ai-thinking-text');
                        }
                        thinkLine.querySelector('.ai-body').innerText += data.token;
                        termContainer.scrollTop = termContainer.scrollHeight;
                    };

                    AIProcess.on('token', tokenListener);

                    AIProcess.query(response._prompt).then(answer => {
                        AIProcess.off('token', tokenListener);
                        if (!firstTokenReceived) {
                            thinkLine.querySelector('.ai-body').classList.remove('ai-thinking-text');
                            thinkLine.querySelector('.ai-body').innerText = answer;
                        }
                        input.disabled = false;
                        input.focus();
                        termContainer.scrollTop = termContainer.scrollHeight;
                    }).catch(() => {
                        AIProcess.off('token', tokenListener);
                        thinkLine.remove();
                        appendAI('*hisses at the error*', 'ai-error');
                        input.disabled = false;
                        input.focus();
                    });
                    return;
                }

                if (response.data) output.innerText += '\n' + response.data;

                // Update prompt
                const newCwd = Shell.getcwd(pid).replace('/home', '~');
                container.querySelector('.prompt').innerText = 'rune:' + newCwd + '$';
                input.value = '';
                termContainer.scrollTop = termContainer.scrollHeight;
            }
        };
    },

    renderProjects: (container) => {
        // stat: the one verified number worth leading with; accent: card colour.
        const projects = [
            { slug: 'sa2', figure: 'hub', title: 'SA2 (CampusFL)', badge: 'Team of 4', accent: '#5ba0f5', featured: true,
              stat: '+9.45 pts', statLabel: 'over synchronous training',
              tags: ['PyTorch', 'Reinforcement Learning', 'FastAPI', 'Android'],
              desc: 'Distributed training across laptops and phones. Time-bounded rounds reach 94.07% vs 84.62% for synchronous training in a 5-minute MNIST run; a Soft Actor-Critic controller matches a tuned heuristic.' },
            { slug: 'awsolotl', figure: 'vault', title: 'AWSolotl', badge: 'Hackathon', accent: '#f0a050', featured: true,
              gh: 'https://github.com/lRunel/AWSolotl', stat: '131 tests', statLabel: '245 assertions',
              tags: ['AWS', 'Cedar', 'DynamoDB', 'pytest'],
              desc: "Checks an AI agent's proposed AWS actions against Cedar policies before they run, and records every decision in an append-only audit log." },
            { slug: 'gem', figure: 'relay', title: 'GEM', accent: '#4eca8b', featured: true,
              gh: 'https://github.com/lRunel/Gem', stat: '5 agents', statLabel: 'coordinated by a supervisor',
              tags: ['Python', 'Gemini API', 'Multi-Agent', 'Speech'],
              desc: 'Voice-driven desktop assistant. A supervisor routes tasks to vision, voice, web, system and knowledge agents that share state through a file-based blackboard.' },
            { slug: 'runeos', figure: 'terminal', title: 'Rune-os', accent: '#b48cf2', gh: 'https://github.com/lRunel/portfolio-website-RuneOS',
              tags: ['C', 'WebAssembly', 'JavaScript'],
              desc: 'This site. A small C kernel with processes, a scheduler, an in-memory filesystem and syscalls, compiled to WebAssembly. The terminal talks to it.' },
            { slug: 'arecanut', figure: 'phone', title: 'Arecanut Disease Detector', badge: 'Team of 4', accent: '#4eca8b',
              tags: ['TensorFlow', 'TFLite', 'React Native'],
              desc: '9-class MobileNetV2 classifier for arecanut crop diseases, 98.5% validation accuracy, INT8-quantized to run offline on phones.' },
            { slug: 'arshirt', figure: 'rail', title: 'ARWear', accent: '#f06aa0',
              tags: ['MediaPipe', 'Three.js', 'Computer Vision'],
              desc: 'Virtual shirt try-on in the browser: 33 tracked body landmarks drive a 3D garment mesh in real time.' },
            { slug: 'linuxflow', figure: 'format', title: 'LinuxFlow-E2B', badge: 'Experimental', accent: '#8888a0',
              tags: ['Python', 'tree-sitter', 'LLM tooling'],
              desc: 'Coding harness for a small Gemma model: retrieves context from a code graph and rejects generated C that breaks simple safety rules before compiling.' },
            { slug: 'skillswap', figure: 'patch', title: 'SkillSwap', badge: 'Hackathon', accent: '#8888a0',
              tags: ['Flask', 'Socket.IO', 'WebRTC'],
              desc: 'Matches people who want to trade skills and connects them for live video sessions.' },
        ];

        const card = (p) => `
            <article class="project-card${p.featured ? ' featured' : ''}" style="--accent:${p.accent}"
                     data-slug="${p.slug}" tabindex="0" role="link" aria-label="${p.title} case study">
                <div class="project-figure"><div data-figure="${p.figure}" data-play="${p.featured ? 'true' : 'false'}"></div></div>
                <div class="project-head">
                    <h3 class="project-title">${p.title}</h3>
                    ${p.badge ? `<span class="research-badge">${p.badge}</span>` : ''}
                </div>
                ${p.stat ? `<div class="project-stat"><span>${p.stat}</span> ${p.statLabel}</div>` : ''}
                <p class="project-desc">${p.desc}</p>
                <div class="project-tags">${p.tags.map(t => `<span>${t}</span>`).join('')}</div>
                <div class="project-foot">
                    <span class="project-more">Case study →</span>
                    ${p.gh ? `<a href="${p.gh}" target="_blank" rel="noopener" class="project-gh">GitHub ↗</a>` : ''}
                </div>
            </article>`;

        container.innerHTML = `
            <div class="projects-container">
                <p class="projects-intro">Featured work first. Move your pointer over the drawings — they react. Every card opens a full write-up.</p>
                <div class="projects-featured">${projects.filter(p => p.featured).map(card).join('')}</div>
                <h4 class="projects-subhead">More projects</h4>
                <div class="projects-grid">${projects.filter(p => !p.featured).map(card).join('')}</div>
            </div>`;

        AppManager.mountFigures(container);

        const open = (el) => window.open(`projects/${el.dataset.slug}/index.html`, '_blank');
        container.querySelectorAll('.project-card').forEach(el => {
            el.addEventListener('click', (e) => { if (!e.target.closest('a')) open(el); });
            el.addEventListener('keydown', (e) => { if (e.key === 'Enter') open(el); });
        });
    },

    renderResume: (container) => {
        const pdf = 'assets/Shubham_Chowta_SWE_Backend.pdf';
        container.innerHTML = `
            <div style="display:flex;flex-direction:column;height:100%">
                <div style="display:flex;justify-content:flex-end;gap:8px;padding:8px 12px;background:#141414;border-bottom:1px solid #2a2a2a">
                    <a href="${pdf}" target="_blank" rel="noopener" class="overview-btn">Open in new tab ↗</a>
                    <a href="${pdf}" download class="overview-btn primary">Download PDF</a>
                </div>
                <iframe src="${pdf}" title="Resume PDF" style="flex:1;width:100%;border:none;background:#1a1a1a"></iframe>
            </div>`;
    },

    renderTextViewer: (container, args, pid) => {
        const out = Shell.read(pid, `/home/${args.filename}`);
        container.innerHTML = `<div class="text-view-container">${out.data}</div>`;
    },

    renderHtmlViewer: (container, args, pid) => {
        const iframe = document.createElement('iframe');
        iframe.style.cssText = 'width:100%;height:100%;border:none;background:#0d0d0d;';
        if (args.vfsPath) {
            // Read project HTML from kernel VFS via syscall
            const out = Shell.read(pid, args.vfsPath);
            iframe.srcdoc = out.data || '<p style="color:#888;padding:32px">Failed to load from VFS</p>';
        } else {
            iframe.src = args.src || 'about:blank';
        }
        container.appendChild(iframe);
    },

    renderSettings: (container, pid) => {
        const kernelInfo = 'v1.0.0 (WASM32-Shell)';
        const procInfo = Shell.ps(pid).data || 'No processes';
        container.innerHTML = `
            <div class="profile-card">
                <div class="profile-section" style="margin-top:0">
                    <div class="profile-section-title">System</div>
                    <div class="profile-edu">
                        <div class="school">Rune-os v1.0</div>
                        <div class="degree">Kernel: Monolithic (C → WebAssembly)</div>
                        <div class="degree">Architecture: WASM32</div>
                        <div class="degree">Scheduler: Round-Robin Cooperative</div>
                    </div>
                </div>
                <div class="profile-section">
                    <div class="profile-section-title">Kernel Info</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#aaa;background:rgba(0,0,0,0.3);padding:12px;border-radius:8px;white-space:pre-wrap;line-height:1.5">${kernelInfo}</div>
                </div>
                <div class="profile-section">
                    <div class="profile-section-title">Processes</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#aaa;background:rgba(0,0,0,0.3);padding:12px;border-radius:8px;white-space:pre-wrap;line-height:1.5">${procInfo}</div>
                </div>
                <div class="profile-section">
                    <div class="profile-section-title">Filesystem</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#aaa;background:rgba(0,0,0,0.3);padding:12px;border-radius:8px;white-space:pre-wrap;line-height:1.5">${Shell.ls(pid, '/').data || '/'}</div>
                </div>
            </div>
        `;
    },

    renderMail: (container) => {
        container.innerHTML = `
            <div class="contact-card">
                <div class="contact-figure"><div data-figure="plug" data-play="true" data-label="A plug being drawn toward a wall socket"></div></div>
                <h3 class="contact-title">Let's connect</h3>
                <p class="contact-desc">Hiring, collaborating, or just curious about the kernel — email is the fastest way to reach me.</p>
                <div class="contact-actions">
                    <a class="overview-btn primary" href="mailto:shubhamchouta@gmail.com">
                        <span class="material-symbols-rounded">send</span> shubhamchouta@gmail.com
                    </a>
                    <a class="overview-btn" href="https://www.linkedin.com/in/shubham-chowta/" target="_blank" rel="noopener">
                        <span class="material-symbols-rounded">link</span> LinkedIn
                    </a>
                    <a class="overview-btn" href="https://github.com/lRunel" target="_blank" rel="noopener">
                        <span class="material-symbols-rounded">code</span> GitHub
                    </a>
                </div>
            </div>`;
        AppManager.mountFigures(container);
    }
};
