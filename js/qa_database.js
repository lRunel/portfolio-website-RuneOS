// qa_database.js — Scripted answers for "Cache", the Rune-os help daemon.
// Rules are checked in order; the first matching pattern wins.

window.QA_RULES = [
    {
        pattern: /\b(hello|hi|hey|greetings)\b/i,
        answer: "Hi! I'm Cache (=^.^=), a scripted helper with fixed answers about Shubham's work. " +
                "Ask about skills, education, contact, or a project: SA2, GEM, AWSolotl, Arecanut, ARWear, RuneOS."
    },
    {
        pattern: /\b(are you|is this) (an? )?(ai|llm|model|chatgpt|real)\b|\bhow do you work\b/i,
        answer: "No language model here — I match your question against a short list of hand-written answers. " +
                "Everything I say is fixed text Shubham wrote."
    },
    {
        pattern: /\b(contact|email|reach|phone|hire|linkedin)\b/i,
        answer: "Email: shubhamchouta@gmail.com · LinkedIn: linkedin.com/in/shubham-chowta · GitHub: github.com/lRunel. " +
                "Type 'download' for the resume PDF."
    },
    {
        pattern: /\b(resume|cv)\b/i,
        answer: "Open Resume from the dock, or type 'download' to save the PDF."
    },
    {
        pattern: /\b(education|college|university|degree|gpa|cgpa|study|studies)\b/i,
        answer: "B.Tech in Computer Science & Engineering (AI & ML) at Sahyadri College of Engineering & Management, " +
                "Mangaluru, 2023–2027. CGPA 8.77/10."
    },
    {
        pattern: /\b(experience|training|scrum|leadership|team)\b/i,
        answer: "Selected as 1 of 60 from 600 applicants for Sahyadri's Software Engineering Training Program (2024–2026). " +
                "Scrum Master for a 5-person team that built and unit-tested a modular banking app in Python, FastAPI and SQLite."
    },
    {
        pattern: /\b(skills?|stack|technolog|languages?)\b/i,
        answer: "Languages: Python, C, C++, SQL, Java, JavaScript. Backend & cloud: FastAPI, Flask, REST, Docker, " +
                "AWS (DynamoDB, S3, IAM, ECS, Bedrock, CDK). ML: PyTorch, TensorFlow/TFLite, reinforcement learning (SAC/PPO). " +
                "Data: Pandas, Polars, KNIME."
    },
    {
        pattern: /\b(sa2|campusfl|distlearn|distributed|federated|fedavg)\b/i,
        answer: "SA2 (CampusFL): distributed training across 3 gaming laptops and 3 Android phones, built by a 4-person team. " +
                "A Soft Actor-Critic scheduler decides how long to wait for slow devices. In a 5-minute MNIST run it reached 94.07% " +
                "test accuracy vs 84.62% for synchronous FedAvg; a tuned fixed heuristic reached 94.36%, so SAC matches the heuristic rather than beating it; the big gain comes from time-bounded rounds."
    },
    {
        pattern: /\b(gem|blackboard|multi-?agent|assistant)\b/i,
        answer: "GEM: a voice-driven desktop assistant in Python. A supervisor routes tasks to 5 sub-agents (vision, voice, web search, " +
                "system, knowledge) that share state through a file-based blackboard. Uses Gemini function calling, faster-whisper STT, " +
                "Piper TTS and a custom wake word. Code: github.com/lRunel/Gem"
    },
    {
        pattern: /\b(awsolotl|lockstep|cedar|aws)\b/i,
        answer: "AWSolotl (Lockstep Recall): a hackathon control plane that checks an AI agent's proposed AWS actions against Cedar policies " +
                "before they run, and logs every decision to an append-only DynamoDB/S3 audit trail. 131 pytest tests (245 assertions). Code: github.com/lRunel/AWSolotl"
    },
    {
        pattern: /\b(arecanut|crop|disease|tflite|drone|uav)\b/i,
        answer: "Arecanut disease detector: a 9-class MobileNetV2 classifier trained on 4,000+ images by a 4-person team, 98.5% validation accuracy. " +
                "INT8 TFLite quantization cut model size by about 70% for offline use in a React Native app."
    },
    {
        pattern: /\b(arwear|ar ?shirt|try-?on|augmented)\b/i,
        answer: "ARWear: browser-based virtual shirt try-on. MediaPipe tracks 33 body landmarks and drives a 3D garment mesh in real time."
    },
    {
        pattern: /\b(linuxflow|gemma)\b/i,
        answer: "LinuxFlow-E2B: an experimental coding harness for the small Gemma model. It retrieves code context from a code graph instead of " +
                "whole files, and rejects generated C that breaks simple rules (function length, nesting, unsafe calls) before compiling it. " +
                "Evaluated on a 25-task benchmark Shubham wrote."
    },
    {
        pattern: /\b(skillswap)\b/i,
        answer: "SkillSwap: a hackathon web app that pairs people who want to trade skills and connects them over WebRTC video. Flask + Socket.IO."
    },
    {
        pattern: /\b(amazon)\b/i,
        answer: "Amazon ML Challenge entry — code: github.com/lRunel/Amazon-ml-challenge-2026"
    },
    {
        pattern: /\b(rune-?os|this (site|website|os)|kernel|wasm|webassembly)\b/i,
        answer: "RuneOS is this site: a small C kernel (processes, round-robin scheduler, in-memory filesystem, syscalls) compiled to " +
                "WebAssembly with Emscripten. The terminal sends your commands to it. Try 'ps' or 'ls /proc'. Code: github.com/lRunel/portfolio-website-RuneOS"
    },
    {
        pattern: /\b(projects?|built|work)\b/i,
        answer: "Projects: RuneOS (this site), SA2, GEM, AWSolotl, Arecanut disease detector, ARWear, LinuxFlow-E2B, SkillSwap. " +
                "Ask about any of them by name, or open Projects from the dock."
    },
    {
        pattern: /\b(thanks|thank you)\b/i,
        answer: "You're welcome! =^.^="
    }
];
