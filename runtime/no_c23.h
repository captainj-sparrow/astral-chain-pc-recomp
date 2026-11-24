#ifndef _GNU_SOURCE
# define _GNU_SOURCE
#endif
#if defined(__linux__) && !defined(__ANDROID__)
# include <features.h>
# undef __GLIBC_USE_C23_STRTOL
# define __GLIBC_USE_C23_STRTOL 0
# undef __GLIBC_USE_ISOC23
# define __GLIBC_USE_ISOC23 0
#endif
