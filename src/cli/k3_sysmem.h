/* k3_sysmem.h - how much memory this machine can actually hand out, on every platform
 * the engine builds on.
 *
 * Two refusals in k3_run.c compare the plan against this number: the KV cache check
 * and the whole-plan check before allocation. Both are guarded by "available > 0", so
 * a probe that returns 0 does not fail loudly, it silently turns both guards off, and
 * --preset auto refuses outright. Until this header the probe was /proc/meminfo behind
 * #ifdef _WIN32, which is exactly 0 on macOS.
 *
 *   Linux     MemAvailable, the kernel's own estimate of what can be handed out without
 *             swapping, which counts reclaimable page cache; MemFree does not.
 *   Darwin    free + inactive + purgeable pages from host_statistics64. Darwin has no
 *             MemAvailable; this is the same set of pages, the ones the kernel will give
 *             up without paging anything out. free_count already includes speculative.
 *   Windows   ullAvailPhys, the MemAvailable equivalent (it accounts for the standby
 *             list the way MemAvailable accounts for cache), capped by ullAvailPageFile,
 *             the remaining commit limit: a machine with a small pagefile can refuse to
 *             commit an allocation while plenty of physical memory sits free, and the
 *             engine's arenas are committed up front.
 *
 * Both functions return 0 when the platform cannot say, which every caller already
 * treats as "unknown", not as "none". */
#ifndef K3_SYSMEM_H
#define K3_SYSMEM_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

#if !defined(_WIN32) && !defined(__APPLE__)
/* One "Field:" line of /proc/meminfo, in bytes; 0 if absent. */
static double k3_meminfo_field(const char *field)
{
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0.0;
    const size_t n = strlen(field);
    char line[256];
    double kb = 0.0;
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, field, n)) { kb = atof(line + n); break; }
    fclose(f);
    return kb * 1024.0;
}
#endif

static double k3_mem_available_bytes(void)
{
#if defined(_WIN32)
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof ms;
    if (!GlobalMemoryStatusEx(&ms)) return 0.0;
    return (double)(ms.ullAvailPhys < ms.ullAvailPageFile ? ms.ullAvailPhys
                                                          : ms.ullAvailPageFile);
#elif defined(__APPLE__)
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_size_t page = 0;
    const mach_port_t host = mach_host_self();
    if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm, &count) != KERN_SUCCESS)
        return 0.0;
    if (host_page_size(host, &page) != KERN_SUCCESS || page == 0) return 0.0;
    return ((double)vm.free_count + (double)vm.inactive_count + (double)vm.purgeable_count)
           * (double)page;
#else
    return k3_meminfo_field("MemAvailable:");
#endif
}

static double k3_mem_total_bytes(void)
{
#if defined(_WIN32)
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof ms;
    if (!GlobalMemoryStatusEx(&ms)) return 0.0;
    return (double)ms.ullTotalPhys;
#elif defined(__APPLE__)
    uint64_t bytes = 0;
    size_t len = sizeof bytes;
    if (sysctlbyname("hw.memsize", &bytes, &len, NULL, 0) != 0) return 0.0;
    return (double)bytes;
#else
    return k3_meminfo_field("MemTotal:");
#endif
}

#endif /* K3_SYSMEM_H */
