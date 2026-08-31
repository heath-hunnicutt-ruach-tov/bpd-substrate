// soa_launch_log.h — host-side logging for SoA kernel launches
#ifndef SOA_LAUNCH_LOG_H
#define SOA_LAUNCH_LOG_H
#include <stdio.h>
static inline void soa_log_launch(int nrows, int ncols, int blocks, long ne00, long ne01, long ncols_dst) {
    static int lc = 0;
    lc++;
    if (lc <= 10) {
        FILE *fp = fopen("/tmp/soa_launch.log", "a");
        if (fp) {
            fprintf(fp, "launch %d: nrows=%d ncols=%d blocks=%d ne00=%ld ne01=%ld ncols_dst=%ld\n",
                    lc, nrows, ncols, blocks, ne00, ne01, ncols_dst);
            fclose(fp);
        }
    }
}
#endif
