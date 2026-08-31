#ifndef SOA_GATE_LOG_H
#define SOA_GATE_LOG_H
#include <stdio.h>
#include <string.h>
static inline void soa_gate_log(const char *name, int op, int fused) {
    static int glc = 0;
    glc++;
    if (glc <= 30) {
        FILE *fp = fopen("/tmp/gate_detect.log", "a");
        if (fp) {
            fprintf(fp, "d%d: name=%s op=%d fused=%d\n", glc, name ? name : "null", op, fused);
            fclose(fp);
        }
    }
}
static inline void soa_swiglu_log(const char *name, int layer, void *up_ptr) {
    static int slc = 0;
    slc++;
    if (slc <= 30) {
        FILE *fp = fopen("/tmp/swiglu.log", "a");
        if (fp) {
            fprintf(fp, "s%d: name=%s layer=%d up_ptr=%p\n", slc, name ? name : "null", layer, up_ptr);
            fclose(fp);
        }
    }
}
#endif
