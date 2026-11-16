/* project_gem.c — short project summary served at ~/projects/gem.txt */

static const char *project_gem_text = "GEM — multi-agent desktop assistant\n===================================\nVoice-driven assistant in Python. A supervisor routes tasks to 5 sub-agents (vision, voice, web, system, knowledge)\nthat share state through a file-based blackboard. Gemini function calling, faster-whisper STT, Piper TTS, custom wake word.\n\nFull write-up: projects/gem/index.html\nCode: https://github.com/lRunel/Gem\n";

const char *get_project_gem_text(void) {
    return project_gem_text;
}
