// macOS implementation of platform.h, files and processes: POSIX files as in linux.c (with F_FULLFSYNC where data must
// reach the disk, since fsync only hands it to the drive), posix_spawn with POSIX_SPAWN_CLOEXEC_DEFAULT for child
// processes, libproc and kqueue in place of /proc and pidfd, CoreFoundation for the locale, CoreText for the CJK
// fonts, Mach and IOKit for the load counters.
#define _DARWIN_C_SOURCE
#include <CoreFoundation/CoreFoundation.h>
#include <CoreText/CoreText.h>
#include <IOKit/IOKitLib.h>
#include <crt_externs.h>
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <libproc.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <mach/mach.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/event.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <SDL3/SDL_clipboard.h>

#include "platform.h"

#define environ (*_NSGetEnviron())

// ---- files

int plat_file_exists(const char *path) {
    struct stat st;
    return !stat(path, &st) && !S_ISDIR(st.st_mode);
}

int plat_dir_exists(const char *path) {
    struct stat st;
    return !stat(path, &st) && S_ISDIR(st.st_mode);
}

int plat_full_path(const char *path, char *out, size_t cap) {
    if (path[0] == '/') return snprintf(out, cap, "%s", path) < (int)cap;
    char cwd[PATH_MAX];
    if (!getcwd(cwd, sizeof cwd)) return 0;
    return snprintf(out, cap, "%s/%s", cwd, path) < (int)cap;
}

void plat_file_mkdir_p(const char *path) {
    char tmp[PATH_MAX];
    if (snprintf(tmp, sizeof tmp, "%s", path) >= (int)sizeof tmp) return;
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/' || *p == '\\') {
            char c = *p;
            *p = 0;
            mkdir(tmp, 0755);
            *p = c;
        }
    }
    mkdir(tmp, 0755);
}

static int write_all(int fd, const void *data, size_t n) {
    const uint8_t *p = data;
    while (n) {
        ssize_t k = write(fd, p, n);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return 0;
        p += k, n -= (size_t)k;
    }
    return 1;
}

// The file's data on the disk: fsync leaves it in the drive's cache, F_FULLFSYNC flushes that too. A file system
// without it (a network share) gets the fsync.
static int full_sync(int fd) { return !fcntl(fd, F_FULLFSYNC) || !fsync(fd); }

uint8_t *plat_file_read(const char *path, uint64_t *size) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return NULL;
    struct stat st;
    uint8_t *buf = NULL;
    if (!fstat(fd, &st) && S_ISREG(st.st_mode) && st.st_size > 0 && st.st_size < (1ll << 30)) {
        buf = malloc((size_t)st.st_size + 1);
        size_t got = 0;
        while (buf && got < (size_t)st.st_size) {
            ssize_t k = read(fd, buf + got, (size_t)st.st_size - got);
            if (k < 0 && errno == EINTR) continue;
            if (k <= 0) break;
            got += (size_t)k;
        }
        if (buf && got != (size_t)st.st_size) {
            free(buf);
            buf = NULL;
        }
        if (buf) {
            buf[st.st_size] = 0;
            *size = (uint64_t)st.st_size;
        }
    }
    close(fd);
    return buf;
}

int plat_file_write(const char *path, const void *data, size_t n) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return 0;
    int ok = write_all(fd, data, n);
    return close(fd) == 0 && ok;
}

// O_APPEND makes each write land whole at the end of the file, from any thread.
PlatAppend *plat_append_open(const char *path) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_APPEND | O_CLOEXEC, 0644);
    return fd < 0 ? NULL : (PlatAppend *)(intptr_t)(fd + 1);
}

int plat_append_write(PlatAppend *f, const void *data, size_t n) {
    int fd = (int)(intptr_t)f - 1;
    ssize_t k;
    do k = write(fd, data, n);
    while (k < 0 && errno == EINTR);
    return k == (ssize_t)n;
}

// Flushes the directory holding path, so a rename into it survives a crash.
static void sync_dir_of(const char *path) {
    char dir[PATH_MAX];
    if (snprintf(dir, sizeof dir, "%s", path) >= (int)sizeof dir) return;
    char *slash = strrchr(dir, '/');
    if (slash == dir) slash[1] = 0;
    else if (slash) *slash = 0;
    else strcpy(dir, ".");
    int fd = open(dir, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd >= 0) {
        fsync(fd);
        close(fd);
    }
}

int plat_file_replace(const char *from, const char *to) {
    if (rename(from, to)) return 0;
    sync_dir_of(to);
    return 1;
}

int plat_file_write_atomic(const char *path, const void *data, size_t n) {
    // A rename replaces a read-only file here (only the folder's permission counts) but not on Windows: a file the
    // player made read-only stays as it is on both.
    if (access(path, W_OK) && errno != ENOENT) return 0;
    char tmp[PATH_MAX + 48];
    int len = snprintf(tmp, sizeof tmp, "%s.%u.%u.tmp", path, plat_process_id(), plat_thread_id());
    if (len < 0 || len >= (int)sizeof tmp) return 0;
    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return 0;
    int ok = write_all(fd, data, n) && full_sync(fd);
    ok = close(fd) == 0 && ok;
    if (ok && !rename(tmp, path)) {
        sync_dir_of(path);
        return 1;
    }
    unlink(tmp);
    return 0;
}

int plat_file_flush(const char *path) {
    int fd = open(path, O_WRONLY | O_CLOEXEC);
    if (fd < 0) return 0;
    int ok = full_sync(fd);
    close(fd);
    return ok;
}

int plat_file_sync(FILE *f) { return !fflush(f) && full_sync(fileno(f)); }

void plat_file_delete(const char *path) { unlink(path); }

int plat_file_truncate(const char *path, uint64_t size) { return truncate(path, (off_t)size) == 0; }

int64_t plat_file_size(const char *path) {
    struct stat st;
    return stat(path, &st) ? -1 : (int64_t)st.st_size;
}

// The owner's write permission stands for Windows' read-only attribute.
int plat_file_set_readonly(const char *path, int on) {
    struct stat st;
    if (stat(path, &st)) return 0;
    mode_t m = on ? st.st_mode & ~(mode_t)0222 : st.st_mode | S_IWUSR;
    return chmod(path, m & 07777) == 0;
}

int plat_file_list(const char *dir, const char *ext, int (*cb)(const char *name, void *ctx), void *ctx) {
    DIR *d = opendir(dir);
    if (!d) return 0;
    size_t el = strlen(ext);
    for (struct dirent *e; (e = readdir(d));) {
        size_t n = strlen(e->d_name);
        if (n < el || strcmp(e->d_name + n - el, ext) || e->d_name[0] == '.') continue;
        if (!cb(e->d_name, ctx)) break;
    }
    closedir(d);
    return 1;
}

PlatFile *plat_file_open(const char *path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    return fd < 0 ? NULL : (PlatFile *)(intptr_t)(fd + 1);
}

// Darwin has no O_DIRECT; F_NOCACHE keeps the reads out of the unified buffer cache.
PlatFile *plat_file_open_unbuffered(const char *path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd >= 0 && fcntl(fd, F_NOCACHE, 1)) {
        close(fd);
        fd = -1;
    }
    return fd < 0 ? NULL : (PlatFile *)(intptr_t)(fd + 1);
}

static int fd_of(PlatFile *f) { return (int)(intptr_t)f - 1; }

void plat_file_close(PlatFile *f) {
    if (f) close(fd_of(f));
}

// The kernel's write into the buffer fails on a read-only page instead of faulting (EFAULT), and the write watch keeps
// clean watched pages read-only (macos_mem.c): a read into guest memory opens and marks its pages first.
int64_t plat_file_pread(PlatFile *f, void *buf, uint32_t n, uint64_t off) {
    plat_mem_will_write((uint64_t)(uintptr_t)buf, n);
    ssize_t k;
    do k = pread(fd_of(f), buf, n, (off_t)off);
    while (k < 0 && errno == EINTR);
    return k >= 0 ? k : errno == EFAULT ? -2 : -1;
}

// The creation time is the file's birth time, as on Windows.
int plat_file_id(PlatFile *f, PlatFileId *out) {
    struct stat st;
    if (fstat(fd_of(f), &st)) return 0;
    *out = (PlatFileId){(uint64_t)st.st_dev, (uint64_t)st.st_ino, (uint64_t)st.st_size,
                        (uint64_t)st.st_mtimespec.tv_sec * 1000000000u + (uint64_t)st.st_mtimespec.tv_nsec,
                        (uint64_t)st.st_birthtimespec.tv_sec * 1000000000u + (uint64_t)st.st_birthtimespec.tv_nsec};
    return 1;
}

const void *plat_file_map(PlatFile *f) {
    struct stat st;
    if (fstat(fd_of(f), &st) || !st.st_size) return NULL;
    void *v = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd_of(f), 0);
    return v == MAP_FAILED ? NULL : v;
}

static int fs_err(int e) {
    switch (e) {
    case ENOENT: case ENOTDIR: return PLAT_FS_NOT_FOUND;
    case EEXIST: return PLAT_FS_EXISTS;
    case ENOTEMPTY: return PLAT_FS_NOT_EMPTY;
    case EBUSY: case ETXTBSY: return PLAT_FS_BUSY;
    default: return PLAT_FS_ERROR;
    }
}

int plat_fs_kind(const char *path) {
    struct stat st;
    if (lstat(path, &st)) return PLAT_FS_MISSING;
    return S_ISLNK(st.st_mode) ? PLAT_FS_LINK : S_ISDIR(st.st_mode) ? PLAT_FS_DIR : PLAT_FS_FILE;
}

int plat_fs_mkdir(const char *path) { return mkdir(path, 0755) ? fs_err(errno) : PLAT_FS_OK; }

int plat_fs_rmdir(const char *path) {
    struct stat st;
    if (!lstat(path, &st) && S_ISLNK(st.st_mode)) return unlink(path) ? fs_err(errno) : PLAT_FS_OK;
    return rmdir(path) ? fs_err(errno) : PLAT_FS_OK;
}

int plat_fs_unlink(const char *path) { return unlink(path) ? fs_err(errno) : PLAT_FS_OK; }

// RENAME_EXCL keeps Windows' rule that a rename never replaces; a file system without it (ENOTSUP) gets a check first,
// which another process could race but the save directories are ours.
int plat_fs_rename(const char *from, const char *to, int durable) {
    if (renamex_np(from, to, RENAME_EXCL)) {
        if (errno != ENOTSUP && errno != EINVAL) return fs_err(errno);
        if (plat_fs_kind(to) != PLAT_FS_MISSING) return PLAT_FS_EXISTS;
        if (rename(from, to)) return fs_err(errno);
    }
    if (durable) {
        sync_dir_of(from);
        sync_dir_of(to);
    }
    return PLAT_FS_OK;
}

int plat_fs_link(const char *existing, const char *path) { return link(existing, path) ? fs_err(errno) : PLAT_FS_OK; }

int plat_fs_links(const char *path) {
    struct stat st;
    return stat(path, &st) ? 0 : (int)st.st_nlink;
}

void plat_fs_sync_dir(const char *path) {
    int fd = open(path, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd >= 0) {
        fsync(fd);
        close(fd);
    }
}

// Open files never stop a rename here.
void plat_fs_busy_reason(const char *dir, char *out, size_t cap) {
    (void)dir;
    if (cap) *out = 0;
}

int plat_fs_list(const char *dir, int (*cb)(const char *name, int kind, uint64_t size, void *ctx), void *ctx) {
    DIR *d = opendir(dir);
    if (!d) return fs_err(errno);
    for (struct dirent *e; (e = readdir(d));) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        struct stat st;
        if (fstatat(dirfd(d), e->d_name, &st, AT_SYMLINK_NOFOLLOW)) continue;
        int kind = S_ISLNK(st.st_mode) ? PLAT_FS_LINK : S_ISDIR(st.st_mode) ? PLAT_FS_DIR : PLAT_FS_FILE;
        if (!cb(e->d_name, kind, kind == PLAT_FS_FILE ? (uint64_t)st.st_size : 0, ctx)) break;
    }
    closedir(d);
    return PLAT_FS_OK;
}

PlatFile *plat_fs_open(const char *path, int write, int *err) {
    struct stat st;
    int fd = open(path, (write ? O_RDWR : O_RDONLY) | O_CLOEXEC);
    if (fd >= 0 && (fstat(fd, &st) || !S_ISREG(st.st_mode))) {
        close(fd);
        fd = -1, errno = ENOENT;
    }
    if (fd < 0 && err) *err = fs_err(errno);
    return fd < 0 ? NULL : (PlatFile *)(intptr_t)(fd + 1);
}

PlatFile *plat_fs_create(const char *path, int *err) {
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0 && err) *err = fs_err(errno);
    return fd < 0 ? NULL : (PlatFile *)(intptr_t)(fd + 1);
}

int64_t plat_file_pwrite(PlatFile *f, const void *buf, uint32_t n, uint64_t off) {
    ssize_t k;
    do k = pwrite(fd_of(f), buf, n, (off_t)off);
    while (k < 0 && errno == EINTR);
    return k;
}

int plat_file_set_size(PlatFile *f, uint64_t size) { return ftruncate(fd_of(f), (off_t)size) ? fs_err(errno) : 0; }

int plat_file_fsync(PlatFile *f) { return full_sync(fd_of(f)); }

// TMPDIR is the per-user temporary directory macOS sets for every login session; confstr gives the same without it.
int plat_temp_dir(char *out, size_t cap) {
    char buf[PATH_MAX];
    const char *t = getenv("TMPDIR");
    if ((!t || !*t) && confstr(_CS_DARWIN_USER_TEMP_DIR, buf, sizeof buf)) t = buf;
    if (!t || !*t) t = "/tmp";
    size_t n = strlen(t);
    while (n > 1 && t[n - 1] == '/') n--;
    return snprintf(out, cap, "%.*s", (int)n, t) < (int)cap;
}

uintptr_t plat_exe_base(void) {
    Dl_info d;
    return dladdr((void *)plat_exe_base, &d) ? (uintptr_t)d.dli_fbase : 0;
}

int plat_module_of(uintptr_t addr, char *name, size_t cap, uintptr_t *base) {
    Dl_info d;
    if (!dladdr((void *)addr, &d) || !d.dli_fname) return 0;
    const char *n = strrchr(d.dli_fname, '/');
    *base = (uintptr_t)d.dli_fbase;
    return snprintf(name, cap, "%s", n ? n + 1 : d.dli_fname) < (int)cap;
}

int plat_exe_path(char *out, size_t cap) {
    char raw[PATH_MAX], real[PATH_MAX];
    uint32_t n = sizeof raw;
    if (_NSGetExecutablePath(raw, &n) || !realpath(raw, real)) return 0;
    return snprintf(out, cap, "%s", real) < (int)cap;
}

size_t plat_getenv(const char *name, char *out, size_t cap) {
    const char *v = getenv(name);
    size_t n = v ? strlen(v) : 0;
    if (!n || n >= cap) return 0;
    memcpy(out, v, n + 1);
    return n;
}

// ---- locale: what the user chose in System Settings (CoreFoundation), as an application started from the Finder has
// no LANG. The language list's first entry is already a BCP-47 tag ("ja-JP"); the region format's identifier is ICU's
// ("en_JP", "de_DE@rg=chzzzz"), whose options are dropped.

static void cf_tag(CFStringRef s, char *tag, size_t cap) {
    tag[0] = 0;
    if (!s || !CFStringGetCString(s, tag, (CFIndex)cap, kCFStringEncodingUTF8)) return;
    tag[strcspn(tag, "@")] = 0;
    for (char *p = tag; *p; p++)
        if (*p == '_') *p = '-';
}

static char ui_lang[32], user_locale[32];

static void ui_lang_init(void) {
    CFArrayRef l = CFLocaleCopyPreferredLanguages();
    if (l && CFArrayGetCount(l)) cf_tag(CFArrayGetValueAtIndex(l, 0), ui_lang, sizeof ui_lang);
    if (l) CFRelease(l);
}

static void user_locale_init(void) {
    CFLocaleRef l = CFLocaleCopyCurrent();
    if (!l) return;
    cf_tag(CFLocaleGetIdentifier(l), user_locale, sizeof user_locale);
    CFRelease(l);
}

const char *plat_ui_language(void) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, ui_lang_init);
    return ui_lang;
}

const char *plat_user_locale(void) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, user_locale_init);
    return user_locale;
}

// The system's CJK fonts by family, through CoreText: their files are not in one place (PingFang lives in a framework's
// resources). A family CoreText does not have resolves to another one, which is not taken.
static int font_file(const char *family, char *out, size_t cap) {
    CFStringRef name = CFStringCreateWithCString(NULL, family, kCFStringEncodingUTF8);
    if (!name) return 0;
    CTFontDescriptorRef d = CTFontDescriptorCreateWithNameAndSize(name, 12);
    CFRelease(name);
    if (!d) return 0;
    int ok = 0;
    CFStringRef fam = CTFontDescriptorCopyAttribute(d, kCTFontFamilyNameAttribute);
    CFURLRef url = CTFontDescriptorCopyAttribute(d, kCTFontURLAttribute);
    char got[128];
    if (fam && url && CFStringGetCString(fam, got, sizeof got, kCFStringEncodingUTF8) && !strcmp(got, family))
        ok = CFURLGetFileSystemRepresentation(url, 1, (UInt8 *)out, (CFIndex)cap);
    if (fam) CFRelease(fam);
    if (url) CFRelease(url);
    CFRelease(d);
    return ok && plat_file_exists(out);
}

int plat_font_path(const char *lang, char *out, size_t cap) {
    static const char *const fonts[][3] = {
        {"ja", "Hiragino Sans", "Hiragino Kaku Gothic ProN"},
        {"ko", "Apple SD Gothic Neo", "AppleGothic"},
        {"zh-Hans", "PingFang SC", "Hiragino Sans GB"},
        {"zh-Hant", "PingFang TC", "Heiti TC"},
    };
    for (size_t i = 0; i < sizeof fonts / sizeof *fonts; i++)
        if (!strcmp(lang, fonts[i][0])) return font_file(fonts[i][1], out, cap) || font_file(fonts[i][2], out, cap);
    return 0;
}

// "macOS 27.2.1 (26B131)"
int plat_os_version(char *out, size_t cap) {
    char ver[32] = "", build[32] = "";
    size_t n = sizeof ver, m = sizeof build;
    if (sysctlbyname("kern.osproductversion", ver, &n, NULL, 0) || !ver[0]) return 0;
    if (sysctlbyname("kern.osversion", build, &m, NULL, 0)) build[0] = 0;
    return snprintf(out, cap, build[0] ? "macOS %s (%s)" : "macOS %s", ver, build) < (int)cap;
}

int plat_is_wine(void) { return 0; }

int plat_stderr_path(char *out, size_t cap) {
    struct stat st;
    char path[MAXPATHLEN];
    if (fstat(2, &st) || !S_ISREG(st.st_mode) || fcntl(2, F_GETPATH, path)) return 0;
    return snprintf(out, cap, "%s", path) < (int)cap;
}

// ---- process

void plat_exit_skip_unload(void) {}   // _exit below runs no library destructors already

// The profile runtime's writer, present only in a build with --pgo-generate (tools/build_game.py): _exit skips the
// atexit handler that would write the .profraw. Looked up, as Mach-O has no weak undefined symbols without linker flags.
void plat_exit(int code) {
    int (*write_profile)(void) = (int (*)(void))dlsym(RTLD_DEFAULT, "__llvm_profile_write_file");
    if (write_profile) write_profile();
    fflush(stdout);
    fflush(stderr);
    _exit(code);
}

int plat_setenv(const char *name, const char *value) {
    return (value ? setenv(name, value, 1) : unsetenv(name)) == 0;
}

uint32_t plat_process_id(void) { return (uint32_t)getpid(); }

uint64_t plat_process_cpu_ns(void) { return clock_gettime_nsec_np(CLOCK_PROCESS_CPUTIME_ID); }

// A process's start, in microseconds since 1970 (libproc); 0 when it is gone.
static uint64_t process_start_us(int pid) {
    struct proc_bsdinfo bi;
    if (proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &bi, sizeof bi) != (int)sizeof bi) return 0;
    return (uint64_t)bi.pbi_start_tvsec * 1000000u + bi.pbi_start_tvusec;
}

uint64_t plat_process_uptime_ns(void) {
    uint64_t start = process_start_us(getpid()), now = plat_unix_time_ns() / 1000u;
    return start && now > start ? (now - start) * 1000u : 0;
}

uint64_t plat_unix_time_ns(void) { return clock_gettime_nsec_np(CLOCK_REALTIME); }

void plat_local_time(struct tm *out) {
    time_t t = time(NULL);
    localtime_r(&t, out);
}

// The range of the Windows CRT's localtime_s and gmtime_s (1970 to 3000): the game's time service reports times
// outside it as invalid on every host.
#define PLAT_TIME_MAX 32535215999ll

int plat_localtime(int64_t unix_s, struct tm *out) {
    time_t t = (time_t)unix_s;
    return unix_s >= 0 && unix_s <= PLAT_TIME_MAX && localtime_r(&t, out) != NULL;
}

int plat_gmtime(int64_t unix_s, struct tm *out) {
    time_t t = (time_t)unix_s;
    return unix_s >= 0 && unix_s <= PLAT_TIME_MAX && gmtime_r(&t, out) != NULL;
}

// The standard offset: of mid-January's and mid-July's, the one further west (summer time moves the clock east).
long plat_tz_west(void) {
    tzset();
    time_t now = time(NULL);
    struct tm tm;
    localtime_r(&now, &tm);
    long west = 0;
    for (int mon = 0; mon < 12; mon += 6) {
        struct tm probe = {.tm_year = tm.tm_year, .tm_mon = mon, .tm_mday = 15, .tm_hour = 12, .tm_isdst = -1};
        time_t t = mktime(&probe);
        struct tm at;
        localtime_r(&t, &at);
        if (!mon || -at.tm_gmtoff > west) west = -at.tm_gmtoff;
    }
    return west;
}

// The executable's arguments as it was started (crt_externs), NULL-terminated.
static char **self_argv(void) { return *_NSGetArgv(); }

void plat_reexec(void) {
    char exe[PATH_MAX];
    fflush(stdout);
    fflush(stderr);
    if (plat_exe_path(exe, sizeof exe)) execv(exe, self_argv());
    fprintf(stderr, "FATAL: re-running the process failed (%s)\n", strerror(errno));
    _exit(70);
}

// Pipes whose ends this process keeps from exec'd children: POSIX_SPAWN_CLOEXEC_DEFAULT gives a child only what its
// file actions name, so no child inherits another one's pipes either.
static int pipe_cloexec(int p[2]) {
    if (pipe(p)) return -1;
    fcntl(p[0], F_SETFD, FD_CLOEXEC);
    fcntl(p[1], F_SETFD, FD_CLOEXEC);
    return 0;
}

static void spawn_attr(posix_spawnattr_t *at, short flags) {
    posix_spawnattr_init(at);
    sigset_t none, all;
    sigemptyset(&none);
    sigfillset(&all);
    posix_spawnattr_setsigmask(at, &none);
    posix_spawnattr_setsigdefault(at, &all);
    posix_spawnattr_setflags(at, flags | POSIX_SPAWN_CLOEXEC_DEFAULT | POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF);
}

int plat_run_self(const char *env_name, const char *env_value, char *out, size_t cap, uint32_t timeout_ms) {
    char exe[PATH_MAX];
    if (!plat_exe_path(exe, sizeof exe)) return -1;
    int p[2];
    if (pipe_cloexec(p)) return -1;
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addinherit_np(&fa, 0);
    posix_spawn_file_actions_adddup2(&fa, p[1], 1);
    posix_spawn_file_actions_adddup2(&fa, p[1], 2);
    posix_spawnattr_t at;
    spawn_attr(&at, 0);
    setenv(env_name, env_value, 1);
    pid_t pid;
    int e = posix_spawn(&pid, exe, &fa, &at, self_argv(), environ);
    unsetenv(env_name);
    posix_spawnattr_destroy(&at);
    posix_spawn_file_actions_destroy(&fa);
    close(p[1]);
    if (e) {
        close(p[0]);
        return -1;
    }
    size_t len = 0;
    char sink[512];
    ssize_t k;
    while (len + 1 < cap && (k = read(p[0], out + len, cap - 1 - len)) != 0)
        if (k > 0) len += (size_t)k;
        else if (errno != EINTR) break;
    if (cap) out[len] = 0;
    while ((k = read(p[0], sink, sizeof sink)) > 0 || (k < 0 && errno == EINTR)) {}
    close(p[0]);
    int status = 0;
    uint64_t end = plat_now_ns() + (uint64_t)timeout_ms * 1000000u;
    while (waitpid(pid, &status, timeout_ms ? WNOHANG : 0) == 0) {
        if (plat_now_ns() >= end) {
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            break;
        }
        plat_sleep_ms(10);
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
}

int plat_random(void *out, size_t n) {
    arc4random_buf(out, n);   // the kernel's generator; cannot fail
    return 1;
}

// ---- desktop

int plat_clipboard_set(const char *utf8) { return SDL_SetClipboardText(utf8); }

static void reap_main(void *arg) { waitpid((pid_t)(intptr_t)arg, NULL, 0); }

// open(1) shows the folder in the Finder; a thread of its own reaps it.
int plat_open_folder(const char *path) {
    struct stat st;
    if (stat(path, &st) || !S_ISDIR(st.st_mode)) return 0;
    posix_spawnattr_t at;
    spawn_attr(&at, POSIX_SPAWN_SETSID);
    char *argv[] = {"/usr/bin/open", (char *)path, NULL};
    pid_t pid;
    int e = posix_spawn(&pid, argv[0], NULL, &at, argv, environ);
    posix_spawnattr_destroy(&at);
    if (e) return 0;
    plat_thread_spawn(reap_main, (void *)(intptr_t)pid, "reap", 64 << 10);
    return 1;
}

// ---- dynamic libraries

void *plat_lib_open(const char *name) { return dlopen(name, RTLD_NOW | RTLD_LOCAL); }
void *plat_lib_sym(void *lib, const char *name) { return lib ? dlsym(lib, name) : NULL; }
const char *plat_lib_error(void) {
    const char *e = dlerror();
    return e ? e : "";
}

// ---- machine load: Mach host statistics for the CPU and memory, the GPU's IOAccelerator PerformanceStatistics (the
// numbers Activity Monitor shows) for the GPU. No CPU temperature: the sensors need private interfaces.

struct PlatLoad {
    uint64_t busy, total, proc_ns, wall_ns;   // the previous call's; total 0 before the first
    int integrated;
};

PlatLoad *plat_load_open(const uint8_t *gpu_luid, int pci_bus, int integrated) {
    (void)gpu_luid, (void)pci_bus;   // Apple Silicon has one GPU
    PlatLoad *l = calloc(1, sizeof *l);
    if (l) l->integrated = integrated;
    return l;
}

void plat_load_close(PlatLoad *l) { free(l); }

void plat_cpu_load(PlatLoad *l, float *total_pct, float *proc_pct, float *temp_c) {
    *total_pct = *proc_pct = *temp_c = -1;
    if (!l) return;
    host_cpu_load_info_data_t c;
    mach_msg_type_number_t n = HOST_CPU_LOAD_INFO_COUNT;
    mach_port_t host = mach_host_self();
    kern_return_t kr = host_statistics(host, HOST_CPU_LOAD_INFO, (host_info_t)&c, &n);
    mach_port_deallocate(mach_task_self(), host);
    if (kr != KERN_SUCCESS) return;
    uint64_t busy = (uint64_t)c.cpu_ticks[CPU_STATE_USER] + c.cpu_ticks[CPU_STATE_SYSTEM] + c.cpu_ticks[CPU_STATE_NICE];
    uint64_t total = busy + c.cpu_ticks[CPU_STATE_IDLE];
    uint64_t pns = plat_process_cpu_ns(), wall = plat_now_ns();
    if (l->total && total > l->total) {
        *total_pct = 100.0f * (float)(busy - l->busy) / (float)(total - l->total);
        *proc_pct = (float)(100.0 * (double)(pns - l->proc_ns) / ((double)(wall - l->wall_ns) * plat_cpu_count()));
    }
    l->busy = busy, l->total = total, l->proc_ns = pns, l->wall_ns = wall;
}

// The first IOAccelerator's PerformanceStatistics number for key; -1 when there is none.
static double accel_stat(const char *key) {
    double v = -1;
    io_iterator_t it;
    if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &it) != KERN_SUCCESS)
        return v;
    for (io_object_t s; v < 0 && (s = IOIteratorNext(it));) {
        CFDictionaryRef d = IORegistryEntryCreateCFProperty(s, CFSTR("PerformanceStatistics"), kCFAllocatorDefault, 0);
        if (d && CFGetTypeID(d) == CFDictionaryGetTypeID()) {
            CFStringRef k = CFStringCreateWithCString(NULL, key, kCFStringEncodingUTF8);
            CFNumberRef num = k ? CFDictionaryGetValue(d, k) : NULL;
            int64_t x;
            if (num && CFGetTypeID(num) == CFNumberGetTypeID() && CFNumberGetValue(num, kCFNumberSInt64Type, &x))
                v = (double)x;
            if (k) CFRelease(k);
        }
        if (d) CFRelease(d);
        IOObjectRelease(s);
    }
    IOObjectRelease(it);
    return v;
}

void plat_gpu_load(PlatLoad *l, float *load_pct, float *vram_used_mb) {
    *load_pct = *vram_used_mb = -1;
    if (!l) return;
    double busy = accel_stat("Device Utilization %"), used = accel_stat("In use system memory");
    if (busy >= 0) *load_pct = (float)busy;
    if (used >= 0) *vram_used_mb = (float)(used / 1048576.0);
}

static uint64_t phys_total(void) {
    uint64_t v = 0;
    size_t n = sizeof v;
    return sysctlbyname("hw.memsize", &v, &n, NULL, 0) ? 0 : v;
}

void plat_mem_info(float *used_mb, float *total_mb, float *proc_mb) {
    *used_mb = *total_mb = *proc_mb = -1;
    uint64_t total = phys_total();
    vm_statistics64_data_t vs;
    mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    mach_port_t host = mach_host_self();
    if (total && host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vs, &n) == KERN_SUCCESS) {
        // in use as Activity Monitor counts it: app memory (internal minus purgeable), wired and compressed
        uint64_t used = ((uint64_t)vs.internal_page_count - vs.purgeable_count + vs.wire_count +
                         vs.compressor_page_count) * (uint64_t)vm_kernel_page_size;
        *total_mb = (float)(total >> 20), *used_mb = (float)(used >> 20);
    }
    mach_port_deallocate(mach_task_self(), host);
    uint64_t resident, private_bytes;
    plat_process_memory(&resident, &private_bytes);
    if (resident) *proc_mb = (float)(resident >> 20);
}

// Resident: the pages in memory now. Private: the physical footprint, the memory the system charges to the process
// (what Activity Monitor shows as its memory), the nearest to Windows' private bytes.
void plat_process_memory(uint64_t *resident, uint64_t *private_bytes) {
    *resident = *private_bytes = 0;
    task_vm_info_data_t vi;
    mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vi, &n) != KERN_SUCCESS) return;
    *resident = vi.resident_size;
    *private_bytes = vi.phys_footprint;
}

// Darwin has no commit limit: swap files grow as needed, and memory pressure, not a limit, ends a process.
void plat_commit_info(double *proc_mb, double *used_mb, double *limit_mb) {
    *proc_mb = *used_mb = *limit_mb = -1;
    uint64_t resident, private_bytes;
    plat_process_memory(&resident, &private_bytes);
    if (private_bytes) *proc_mb = (double)(private_bytes >> 20);
}

// ---- child processes

struct PlatChild {
    pid_t pid;
    int in, out;   // in: our write end, out: our read end
};

PlatChild *plat_child_start(const char *exe, const char *arg, uint32_t *err) {
    if (err) *err = 0;
    int in[2], out[2];
    if (pipe_cloexec(in)) return NULL;
    if (pipe_cloexec(out)) {
        close(in[0]), close(in[1]);
        return NULL;
    }
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, in[0], 0);
    posix_spawn_file_actions_adddup2(&fa, out[1], 1);
    posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
    posix_spawnattr_t at;
    spawn_attr(&at, 0);
    char *argv[] = {(char *)exe, (char *)arg, NULL};
    pid_t pid;
    int e = posix_spawn(&pid, exe, &fa, &at, argv, environ);
    posix_spawnattr_destroy(&at);
    posix_spawn_file_actions_destroy(&fa);
    close(in[0]);
    close(out[1]);
    if (e) {
        close(in[1]), close(out[0]);
        if (err) *err = (uint32_t)e;
        return NULL;
    }
    PlatChild *c = malloc(sizeof *c);
    if (!c) {
        kill(pid, SIGKILL);
        waitpid(pid, NULL, 0);
        close(in[1]), close(out[0]);
        return NULL;
    }
    c->pid = pid, c->in = in[1], c->out = out[0];
    return c;
}

void plat_child_stop(PlatChild *c) {
    if (!c) return;
    close(c->in);
    close(c->out);
    kill(c->pid, SIGKILL);
    waitpid(c->pid, NULL, 0);
    free(c);
}

// A child that died makes the write fail with EPIPE; SIGPIPE is ignored for the process (main.c).
int plat_child_write(PlatChild *c, const void *data, size_t n) { return write_all(c->in, data, n); }

long plat_child_read(PlatChild *c, void *buf, size_t cap) {
    ssize_t k;
    do k = read(c->out, buf, cap);
    while (k < 0 && errno == EINTR);
    return (long)k;
}

// ---- restart
// AC_RESTART_WAIT names the process that called plat_restart, as its id and start time (libproc, microseconds since
// 1970): a process that has only reused the id is not waited for. Waiting keeps the new process off the old one's files
// (the log it moves at start, the saves) until the old one is gone. The new process gets stdin, stdout and stderr and
// no other descriptor, and the default signal mask and dispositions.
#define RESTART_WAIT "AC_RESTART_WAIT"

int plat_restart(void) {
    char exe[PATH_MAX], wait[64];
    if (!plat_exe_path(exe, sizeof exe)) return 0;
    snprintf(wait, sizeof wait, "%d.%llu", (int)getpid(), (unsigned long long)process_start_us(getpid()));
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    for (int fd = 0; fd < 3; fd++) posix_spawn_file_actions_addinherit_np(&fa, fd);
    posix_spawnattr_t at;
    spawn_attr(&at, 0);
    setenv(RESTART_WAIT, wait, 1);
    pid_t pid;
    int e = posix_spawn(&pid, exe, &fa, &at, self_argv(), environ);
    unsetenv(RESTART_WAIT);
    posix_spawnattr_destroy(&at);
    posix_spawn_file_actions_destroy(&fa);
    return e == 0;
}

int plat_spawn(const char *const *argv, const char *cwd) {
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    for (int fd = 0; fd < 3; fd++) posix_spawn_file_actions_addinherit_np(&fa, fd);
    if (__builtin_available(macOS 26.0, *)) {
        posix_spawn_file_actions_addchdir(&fa, cwd);
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        posix_spawn_file_actions_addchdir_np(&fa, cwd);
#pragma clang diagnostic pop
    }
    posix_spawnattr_t at;
    spawn_attr(&at, POSIX_SPAWN_SETSID);
    pid_t pid;
    int e = posix_spawn(&pid, argv[0], &fa, &at, (char *const *)argv, environ);
    posix_spawnattr_destroy(&at);
    posix_spawn_file_actions_destroy(&fa);
    return e == 0;
}

int plat_restart_wait(uint32_t timeout_ms) {
    const char *v = getenv(RESTART_WAIT);
    if (!v) return 0;
    int pid = 0;
    unsigned long long start = 0;
    int ok = sscanf(v, "%d.%llu", &pid, &start) == 2 && pid > 0 && start;
    unsetenv(RESTART_WAIT);   // the processes this one starts do not wait again
    if (!ok) return 0;
    // The exit watch is registered first: a start time that still matches then proves it is that process, not a new
    // one with the id, and its exit cannot slip in between.
    int kq = kqueue();
    if (kq < 0) return 0;
    struct kevent ev;
    EV_SET(&ev, (uintptr_t)pid, EVFILT_PROC, EV_ADD | EV_ONESHOT, NOTE_EXIT, 0, NULL);
    int r = 0;
    if (!kevent(kq, &ev, 1, NULL, 0, NULL) && process_start_us(pid) == start) {
        struct timespec ts = {(time_t)(timeout_ms / 1000), (long)(timeout_ms % 1000) * 1000000};
        int k;
        do k = kevent(kq, NULL, 0, &ev, 1, &ts);
        while (k < 0 && errno == EINTR);
        r = k > 0 ? 1 : -1;
    }
    close(kq);   // registering fails with ESRCH when the process is gone: nothing to wait for
    return r;
}
