/* project_sa2.c — short project summary served at ~/projects/sa2.txt */

static const char *project_sa2_text = "SA2 (CampusFL) — distributed training on mixed devices\n======================================================\nTraining across 3 gaming laptops and 3 Android phones, built by a 4-person team.\nRounds are time-bounded, slow devices get smaller tasks, and stale updates are dropped.\nA Soft Actor-Critic agent picks the wait time, sparsity and staleness limit each round.\n5-minute MNIST run: 94.07% vs 84.62% for synchronous training (+9.45 pts).\nA tuned fixed heuristic reached 94.36%, so SAC matches it rather than beating it.\n\nFull write-up: projects/sa2/index.html\n";

const char *get_project_sa2_text(void) {
    return project_sa2_text;
}
