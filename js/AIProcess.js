/**
 * AIProcess.js — RuneOS AI Daemon (PID 4)
 * 
 * User-space process named "Cache" (the cat =^.^=).
 * Communicates with the C kernel through Shell syscalls.
 * Runs SmolLM2-135M-Instruct inference in a Web Worker.
 * Streams tokens back for real-time display.
 * 
 * Lifecycle:
 *   UNSTARTED → DOWNLOADING → READY
 */
const AIProcess = (() => {
    let _pid = null;
    let _state = 'UNSTARTED';
    let _worker = null;
    let _listeners = {};
    let _ready = false;
    let _currentStage = '';
    let _downloadProgress = { progress: 0, loaded: 0, total: 0 };

    const SYSTEM_PROMPT =
        'You are Cache, a helpful AI cat daemon inside RuneOS built by Shubham. ' +
        'Answer questions clearly and concisely based on your knowledge base. ' +
        'Keep answers very short. You can meow occasionally.';

    function _emit(event, data) {
        const handlers = _listeners[event];
        if (handlers) {
            handlers.forEach(fn => {
                try { fn(data); } catch (e) { console.error('[AI] listener error:', e); }
            });
        }
    }

    function _on(event, fn) {
        if (!_listeners[event]) _listeners[event] = [];
        _listeners[event].push(fn);
    }

    function _off(event, fn) {
        if (_listeners[event]) {
            _listeners[event] = _listeners[event].filter(f => f !== fn);
        }
    }

    function _writeStatus(status) {
        try { Shell.write(_pid, '/ai/status', status); } catch (e) { }
    }

    function _setKernelState(stateName) {
        try { Shell.setstate(_pid, _pid, stateName); } catch (e) { }
    }

    function _initWorker() {
        // Mock instant initialization
        setTimeout(() => {
            _state = 'READY';
            _ready = true;
            _currentStage = 'ready';
            _writeStatus('ready');
            _setKernelState('running');
            _emit('ready', { pid: _pid });
        }, 100);
    }

    return {
        boot() {
            if (_state !== 'UNSTARTED') return;
            _state = 'SPAWNING';

            try {
                _pid = Shell.spawn('ai');
            } catch (e) {
                console.error('[AI] Failed to spawn:', e);
                return;
            }

            if (!_pid || _pid < 0) {
                console.error('[AI] Invalid PID:', _pid);
                return;
            }

            _setKernelState('initializing');
            _writeStatus('downloading');
            _initWorker();
        },

        /**
         * Query Cache. Returns a Promise that resolves when generation is done.
         * Subscribe to 'token' events for streaming.
         */
        query(prompt) {
            return new Promise((resolve) => {
                if (!_ready) {
                    resolve('Cache is still loading... try again in a moment.');
                    return;
                }

                let response = "";
                const lowerPrompt = prompt.toLowerCase();
                const promptWords = lowerPrompt.match(/\b[a-z]{3,}\b/g) || [];
                
                // Helper: Levenshtein distance for fuzzy matching typos
                const levenshtein = (a, b) => {
                    if (a.length === 0) return b.length;
                    if (b.length === 0) return a.length;
                    const matrix = Array.from({ length: a.length + 1 }, () => Array(b.length + 1).fill(0));
                    for (let i = 0; i <= a.length; i++) matrix[i][0] = i;
                    for (let j = 0; j <= b.length; j++) matrix[0][j] = j;
                    for (let i = 1; i <= a.length; i++) {
                        for (let j = 1; j <= b.length; j++) {
                            const cost = a[i - 1] === b[j - 1] ? 0 : 1;
                            matrix[i][j] = Math.min(
                                matrix[i - 1][j] + 1,
                                matrix[i][j - 1] + 1,
                                matrix[i - 1][j - 1] + cost
                            );
                        }
                    }
                    return matrix[a.length][b.length];
                };

                let bestMatch = null;
                let maxScore = 0;

                if (window.QA_DATABASE && promptWords.length > 0) {
                    for (const item of QA_DATABASE) {
                        let score = 0;
                        for (const kw of item.keywords) {
                            for (const pw of promptWords) {
                                if (pw === kw) {
                                    score += 3; // Exact match
                                } else if (kw.includes(pw) || pw.includes(kw)) {
                                    score += 1.5; // Stem/substring match
                                } else if (pw.length > 4 && kw.length > 4 && levenshtein(pw, kw) <= 2) {
                                    score += 1; // Fuzzy typo match
                                }
                            }
                        }
                        
                        score += item.answer.length * 0.001; // tie-breaker for longer, richer context
                        if (score > maxScore && score > 2.5) { // Threshold to prevent garbage matches
                            maxScore = score;
                            bestMatch = item;
                        }
                    }
                }

                // Psychology Wrappers
                const ackWrappers = [
                    "That's an interesting question.",
                    "Ah, let me think about that.",
                    "Good question.",
                    "Here's what I found in my knowledge base:",
                    "Right, concerning that topic:"
                ];
                
                const engageWrappers = [
                    "Does that make sense?",
                    "Let me know if you need more details.",
                    "I hope that clarifies it!",
                    "Is there anything else you want to know?"
                ];

                const fallbackWrappers = [
                    "I'm sorry, I couldn't find specific details on that in the documentation.",
                    "Hmm, that doesn't seem to match my current knowledge base.",
                    "I don't have an exact answer for that right now."
                ];
                
                const fallbackSuggestions = " But I am quite knowledgeable about RuneOS, the Distributed Deep Learning framework, and the Arecanut UAV project. Could I help you with one of those?";

                const random = (arr) => arr[Math.floor(Math.random() * arr.length)];
                
                // Intent recognition
                if (lowerPrompt.match(/\b(hello|hi|hey|greetings)\b/)) {
                    response = "Hello there! I'm Cache, the RuneOS daemon. How can I help you today?";
                } else if (lowerPrompt.match(/\b(thanks|thank you|cool|awesome)\b/)) {
                    response = "You're welcome! Let me know if there's anything else.";
                } else if (bestMatch) {
                    response = `${random(ackWrappers)} ${bestMatch.answer} ${random(engageWrappers)}`;
                } else {
                    response = `${random(fallbackWrappers)}${fallbackSuggestions}`;
                }

                // Simulate token streaming
                const tokens = response.split(' ');
                let currentIndex = 0;

                const streamInterval = setInterval(() => {
                    if (currentIndex < tokens.length) {
                        // Append space back since we split by space
                        _emit('token', { token: tokens[currentIndex] + ' ' });
                        currentIndex++;
                    } else {
                        clearInterval(streamInterval);
                        _emit('result', { text: response });
                        _emit('done', {});
                        resolve(response);
                    }
                }, 50); // 50ms per word simulates fast typing
            });
        },

        isReady() { return _ready; },
        getState() { return _state; },
        getStage() { return _currentStage; },
        getProgress() { return { ..._downloadProgress }; },
        getPid() { return _pid; },
        on: _on,
        off: _off
    };
})();

window.AIProcess = AIProcess;
