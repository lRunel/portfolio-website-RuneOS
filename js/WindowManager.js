const WindowManager = {
            startDrag: (e, header) => {
                if (window.innerWidth <= 768) return;
                if (e.target.closest('button') || e.target.closest('.back-btn')) return;

                const win = header.parentElement;
                AppManager.focus(win);

                // If maximized, unmaximize on drag
                if (win.classList.contains('maximized')) return;

                const shiftX = e.clientX - win.getBoundingClientRect().left;
                const shiftY = e.clientY - win.getBoundingClientRect().top;

                // Disable iframe pointer events during drag so cursor is not captured
                const iframes = document.querySelectorAll('iframe');
                iframes.forEach(f => { f.style.pointerEvents = 'none'; });

                const onPointerMove = (ev) => {
                    let left = ev.clientX - shiftX;
                    let top = ev.clientY - shiftY;
                    const width = win.offsetWidth || parseInt(win.style.width) || 640;
                    const height = win.offsetHeight || parseInt(win.style.height) || 480;
                    const dockHeight = 88;

                    const maxLeft = Math.max(0, window.innerWidth - width);
                    if (left < 0) left = 0;
                    if (left > window.innerWidth - width) left = maxLeft;
                    left = Math.max(0, Math.min(left, maxLeft));

                    const maxTop = Math.max(0, window.innerHeight - dockHeight - height);
                    if (top < 0) top = 0;
                    if (top > window.innerHeight - dockHeight - height) top = maxTop;
                    top = Math.max(0, Math.min(top, maxTop));

                    win.style.left = left + 'px';
                    win.style.top = top + 'px';
                };

                const onPointerUp = (ev) => {
                    if (header.releasePointerCapture && ev && ev.pointerId !== undefined) {
                        try { header.releasePointerCapture(ev.pointerId); } catch (err) {}
                    }
                    window.removeEventListener('pointermove', onPointerMove);
                    window.removeEventListener('pointerup', onPointerUp);
                    window.removeEventListener('pointercancel', onPointerUp);
                    window.removeEventListener('blur', onPointerUp);
                    window.removeEventListener('resize', onPointerUp);
                    iframes.forEach(f => { f.style.pointerEvents = ''; });
                };

                if (header.setPointerCapture && e.pointerId !== undefined) {
                    try { header.setPointerCapture(e.pointerId); } catch (err) {}
                }

                window.addEventListener('pointermove', onPointerMove);
                window.addEventListener('pointerup', onPointerUp);
                window.addEventListener('pointercancel', onPointerUp);
                window.addEventListener('blur', onPointerUp);
                window.addEventListener('resize', onPointerUp);
            },

            toggleMaximize: (btn) => {
                const win = btn.closest('.window');
                if (!win) return;

                const icon = btn.querySelector('.material-symbols-rounded');

                if (win.classList.contains('maximized')) {
                    win.classList.remove('maximized');
                    win.style.left = win._prevLeft || '';
                    win.style.top = win._prevTop || '';
                    win.style.width = win._prevWidth || '';
                    win.style.height = win._prevHeight || '';
                    if (icon) icon.textContent = 'check_box_outline_blank';
                } else {
                    win._prevLeft = win.style.left;
                    win._prevTop = win.style.top;
                    win._prevWidth = win.style.width;
                    win._prevHeight = win.style.height;
                    win.classList.add('maximized');
                    if (icon) icon.textContent = 'filter_none';
                }
            },

            toggleMinimize: (btn) => {
                const win = btn.closest('.window');
                if (!win) return;
                win.classList.add('minimized');
                win.classList.remove('active');
            },

            restoreMinimized: (appName) => {
                const entry = AppManager.instances[appName];
                if (entry && entry.win && entry.win.classList.contains('minimized')) {
                    entry.win.classList.remove('minimized');
                    AppManager.focus(entry.win);
                    return true;
                }
                return false;
            },

            clampOpenWindows: () => {
                if (typeof window === 'undefined' || window.innerWidth <= 768) return;
                const dockHeight = 88;
                if (!AppManager || !AppManager.instances) return;
                Object.values(AppManager.instances).forEach(entry => {
                    const win = entry && entry.win;
                    if (!win || win.classList.contains('maximized') || win.classList.contains('minimized')) return;
                    const width = win.offsetWidth || parseInt(win.style.width) || 640;
                    const height = win.offsetHeight || parseInt(win.style.height) || 480;
                    const maxLeft = Math.max(0, window.innerWidth - width);
                    const maxTop = Math.max(0, window.innerHeight - dockHeight - height);
                    let left = parseInt(win.style.left) || 0;
                    let top = parseInt(win.style.top) || 0;
                    left = Math.max(0, Math.min(left, maxLeft));
                    top = Math.max(0, Math.min(top, maxTop));
                    win.style.left = left + 'px';
                    win.style.top = top + 'px';
                });
            }
        };

        if (typeof window !== 'undefined') {
            window.addEventListener('resize', () => WindowManager.clampOpenWindows());
        }
