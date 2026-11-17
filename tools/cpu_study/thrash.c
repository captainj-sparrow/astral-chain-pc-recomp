// Cache thrasher for the cache-starved bench: each thread walks its own buffer, reading and writing one byte per
// cache line, so the shared last-level cache keeps filling with its lines and the game's lines leave it.
// usage: thrash <threads> <MiB per thread> <affinity mask hex> <seconds>
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static size_t g_bytes;
static volatile LONG g_stop;

static DWORD WINAPI walk(void *arg) {
    (void)arg;
    unsigned char *p = VirtualAlloc(NULL, g_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!p) return 1;
    uint64_t sum = 0;
    while (!g_stop)
        for (size_t i = 0; i < g_bytes; i += 64) sum += p[i]++;
    return (DWORD)sum;
}

int main(int argc, char **argv) {
    if (argc < 5) return 2;
    int n = atoi(argv[1]);
    g_bytes = (size_t)atoi(argv[2]) << 20;
    SetProcessAffinityMask(GetCurrentProcess(), (DWORD_PTR)strtoull(argv[3], NULL, 16));
    SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);
    HANDLE h[64];
    for (int i = 0; i < n && i < 64; i++) h[i] = CreateThread(NULL, 0, walk, NULL, 0, NULL);
    Sleep((DWORD)atoi(argv[4]) * 1000);
    g_stop = 1;
    WaitForMultipleObjects((DWORD)n, h, TRUE, 5000);
    return 0;
}
