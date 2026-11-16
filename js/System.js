const System = {
            booted: false,
            zIndex: 100,

            boot: () => {
                const env = document.getElementById('environment');
                if (env) env.style.visibility = 'visible';

                const onDesktopReady = () => {
                    System.booted = true;
                    System.startLoops();
                    if (window.AIProcess) AIProcess.boot();
                    // Shareable deep link, e.g. ?open=projects
                    const requested = new URLSearchParams(location.search).get('open');
                    const linkable = ['overview', 'projects', 'resume', 'terminal', 'mail'];
                    if (linkable.includes(requested)) {
                        AppManager.launch(requested);
                    } else if (window.innerWidth > 768 && !AppManager.instances['overview']) {
                        // Desktop visitors land on the overview; mobile shows its own summary card.
                        AppManager.launch('overview');
                    }
                };

                KernelModule().then(mod => {
                    _wasmModule = mod;
                    Shell.init(mod);
                    mod.cwrap('wasm_kernel_init', null, [])();
                }).catch(err => {
                    console.warn('[WASM] kernel failed to load:', err);
                }).finally(onDesktopReady);
            },

            startLoops: () => {
                // Clock — updates both mobile status bar and desktop clock widget
                const updateClocks = () => {
                    const now = new Date();
                    const time = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                    const mobileClock = document.getElementById('mobile-clock');
                    if (mobileClock) mobileClock.textContent = time;

                    // Desktop clock widget
                    const clockTime = document.getElementById('clock-time');
                    const clockDate = document.getElementById('clock-date');
                    const clockGreeting = document.getElementById('clock-greeting');
                    if (clockTime) clockTime.textContent = time;
                    if (clockDate) {
                        clockDate.textContent = now.toLocaleDateString([], {
                            weekday: 'long', month: 'long', day: 'numeric'
                        });
                    }
                    if (clockGreeting) {
                        const hour = now.getHours();
                        let greeting = '';
                        if (hour < 5) greeting = 'Burning the midnight oil';
                        else if (hour < 12) greeting = 'Good morning';
                        else if (hour < 17) greeting = 'Good afternoon';
                        else if (hour < 21) greeting = 'Good evening';
                        else greeting = 'Night owl mode';
                        clockGreeting.textContent = greeting;
                    }
                };
                updateClocks();
                setInterval(updateClocks, 1000);

                // Kernel Poll — real WASM kernel state
                const bootedAt = Date.now();
                const widget = document.getElementById('sys-widget');
                const updateWidget = () => {
                    const raw = Shell.read(1, '/kernel/state').data || '';
                    const state = {};
                    raw.split('\n').forEach(line => {
                        const i = line.indexOf(':');
                        if (i > 0) state[line.slice(0, i).trim()] = line.slice(i + 1).trim();
                    });
                    const secs = Math.floor((Date.now() - bootedAt) / 1000);
                    const uptime = secs < 60 ? `${secs}s` : `${Math.floor(secs / 60)}m ${secs % 60}s`;
                    const ok = raw !== '';
                    widget.innerHTML = `
                        <div class="sw-title"><span class="sw-dot${ok ? '' : ' off'}"></span>C kernel · WebAssembly</div>
                        <div class="sw-row"><span>status</span><span>${ok ? 'running' : 'offline'}</span></div>
                        <div class="sw-row"><span>processes</span><span>${state.processes || '–'}</span></div>
                        <div class="sw-row"><span>uptime</span><span>${uptime}</span></div>
                        <button class="sw-cta" type="button">Open terminal →</button>`;
                };
                widget.addEventListener('click', (e) => {
                    if (e.target.closest('.sw-cta')) AppManager.launch('terminal');
                });
                updateWidget();
                setInterval(updateWidget, 1000);

                // Desktop UI Refresher connecting real-time to /home
                let lastDesktopState = "";
                setInterval(() => {
                    const lsData = Shell.ls(1, '/home').data;
                    if (lsData !== undefined && lsData !== lastDesktopState) {
                        lastDesktopState = lsData;
                        const items = lsData.split('\n').filter(i => i.trim() !== '');
                        
                        const ws = document.getElementById('workspace');
                        // clear existing file-icon-items but keep windows securely
                        ws.querySelectorAll('.file-icon-item').forEach(e => e.remove());
                        
                        let offset = 20;
                        for (const item of items) {
                            if (item === 'resume.txt' || item === 'projects') continue; // already reachable from the dock
                            const isFolder = !item.includes('.');
                            const iconStr = isFolder ? 'folder' : 'insert_drive_file';
                            const colorClass = isFolder ? 'color-folder' : 'color-file';
                            
                            const div = document.createElement('div');
                            div.className = 'file-icon-item desktop-item';
                            div.style.position = 'absolute';
                            div.style.top = offset + 'px';
                            div.style.right = '20px';
                            div.style.width = '80px';
                            div.style.cursor = 'pointer';
                            div.innerHTML = `
                                <span class="material-symbols-rounded f-icon ${colorClass}">${iconStr}</span>
                                <span class="f-name" style="color:white;text-shadow:0 1px 3px rgba(0,0,0,0.8);">${item}</span>
                            `;
                            ws.appendChild(div);
                            offset += 100;
                        }
                    }
                }, 1000);

                // Dynamic Mobile Status Bar
                const updateMobileStatus = async () => {
                    const wifiIcon = document.querySelector('.mobile-status-right .material-symbols-rounded:nth-child(2)');
                    const batteryIcon = document.querySelector('.mobile-status-right .material-symbols-rounded:nth-child(3)');
                    
                    if (navigator.connection) {
                        if (!navigator.onLine) {
                            wifiIcon.textContent = 'wifi_off';
                        } else {
                            wifiIcon.textContent = 'wifi';
                        }
                    }
                    
                    if (navigator.getBattery) {
                        try {
                            const battery = await navigator.getBattery();
                            if (battery.charging) {
                                batteryIcon.textContent = 'battery_charging_full';
                            } else if (battery.level > 0.9) {
                                batteryIcon.textContent = 'battery_full';
                            } else if (battery.level > 0.6) {
                                batteryIcon.textContent = 'battery_5_bar';
                            } else if (battery.level > 0.3) {
                                batteryIcon.textContent = 'battery_3_bar';
                            } else {
                                batteryIcon.textContent = 'battery_1_bar';
                            }
                        } catch (e) {}
                    }
                };
                updateMobileStatus();
                setInterval(updateMobileStatus, 60000);
                window.addEventListener('online', updateMobileStatus);
                window.addEventListener('offline', updateMobileStatus);
            }
        };
