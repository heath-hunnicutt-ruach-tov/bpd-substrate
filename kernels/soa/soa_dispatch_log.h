// soa_dispatch_log.h — log dispatch decisions for debugging
#ifndef SOA_DISPATCH_LOG_H
#define SOA_DISPATCH_LOG_H
#include <stdio.h>
static inline void soa_dispatch_log(int type, const void *ids, long ncols_dst, long ne01, int soa_env) {
    static int dlog = 0;
    dlog++;
    if (dlog <= 20) {
        FILE *fp = fopen("/tmp/dispatch.log", "a");
        if (fp) {
            fprintf(fp, "d%d: type=%d ids=%p ncols=%ld ne01=%ld soa=%d\n",
                    dlog, type, ids, ncols_dst, ne01, soa_env);
            fclose(fp);
        }
    }
}
#endif
