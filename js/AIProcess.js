/**
 * AIProcess.js — "Cache", the Rune-os help daemon (=^.^=)
 *
 * A scripted Q&A assistant: questions are matched against the hand-written
 * rules in qa_database.js. There is no language model involved — answers are
 * fixed text about Shubham's work. It is registered as a kernel process so it
 * shows up in `ps` and in /kernel/state like any other process.
 */
const AIProcess = (() => {
    let _pid = null;
    let _state = 'UNSTARTED';
    const _listeners = {};

    function _emit(event, data) {
        (_listeners[event] || []).forEach(fn => {
            try { fn(data); } catch (e) { console.error('[Cache] listener error:', e); }
        });
    }

    function _on(event, fn) {
        (_listeners[event] = _listeners[event] || []).push(fn);
    }

    function _off(event, fn) {
        if (_listeners[event]) _listeners[event] = _listeners[event].filter(f => f !== fn);
    }

    function _answer(prompt) {
        for (const rule of (window.QA_RULES || [])) {
            if (rule.pattern.test(prompt)) return rule.answer;
        }
        return "I only know a fixed set of answers about Shubham's work. Try: 'skills', 'education', " +
               "'contact', 'SA2', 'GEM', 'AWSolotl', 'Arecanut', or 'RuneOS'.";
    }

    return {
        boot() {
            if (_state !== 'UNSTARTED') return;
            try {
                _pid = Shell.spawn('ai');
                Shell.setstate(_pid, _pid, 'running');
                Shell.write(_pid, '/ai/status', 'ready');
            } catch (e) {
                console.warn('[Cache] kernel registration failed:', e);
            }
            _state = 'READY';
            _emit('ready', { pid: _pid });
        },

        /** Resolves with the answer; emits 'token' events word by word for display. */
        query(prompt) {
            return new Promise((resolve) => {
                const response = _answer(prompt);
                const words = response.split(' ');
                let i = 0;
                const timer = setInterval(() => {
                    if (i < words.length) {
                        _emit('token', { token: words[i++] + ' ' });
                    } else {
                        clearInterval(timer);
                        resolve(response);
                    }
                }, 20);
            });
        },

        isReady() { return _state === 'READY'; },
        getState() { return _state; },
        getPid() { return _pid; },
        on: _on,
        off: _off
    };
})();

window.AIProcess = AIProcess;
