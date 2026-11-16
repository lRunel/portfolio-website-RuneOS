/* project_runeos.c — short project summary served at ~/projects/runeos.txt */

static const char *project_runeos_text = "Rune-os — this site\n===================\nA small C kernel (processes, round-robin scheduler, in-memory filesystem, syscalls) compiled to WebAssembly\nwith Emscripten. You are talking to it right now through the terminal.\n\nFull write-up: projects/runeos/index.html\nCode: https://github.com/lRunel/portfolio-website-RuneOS\n";

const char *get_project_runeos_text(void) {
    return project_runeos_text;
}
