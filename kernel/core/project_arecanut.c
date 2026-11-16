/* project_arecanut.c — short project summary served at ~/projects/arecanut.txt */

static const char *project_arecanut_text = "Arecanut Disease Detector\n=========================\n9-class MobileNetV2 classifier for arecanut crop diseases, trained on 4,000+ images by a 4-person team.\n98.5% validation accuracy. INT8 TFLite quantization cut model size by about 70% for offline use in a React Native app.\n\nFull write-up: projects/arecanut/index.html\n";

const char *get_project_arecanut_text(void) {
    return project_arecanut_text;
}
