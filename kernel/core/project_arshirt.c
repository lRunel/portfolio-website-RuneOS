/* project_arshirt.c — short project summary served at ~/projects/arshirt.txt */

static const char *project_arshirt_text = "ARWear — browser virtual try-on\n===============================\nMediaPipe tracks 33 body landmarks and drives a 3D shirt mesh in real time, entirely in the browser.\n\nFull write-up: projects/arshirt/index.html\n";

const char *get_project_arshirt_text(void) {
    return project_arshirt_text;
}
