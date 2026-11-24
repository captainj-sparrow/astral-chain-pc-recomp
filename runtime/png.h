// PNG writer shared by the flight recorder's thumbnails (flight.c) and the photo mode's screenshots (photo.c):
// 8-bit RGB, every row Paeth-filtered, one fixed-Huffman deflate block. No compression library is linked.
#pragma once
#include <stddef.h>
#include <stdint.h>

// Work buffers, grown on demand and kept between calls of one thread; png_bufs_free releases them.
typedef struct {
    uint8_t *raw, *z;
    int32_t *head;
    size_t cap;   // bytes of raw; z holds cap * 9 / 8 + 64
} PngBufs;

// px: h rows of w pixels, `step` bytes each (3: RGB, 4: RGBA or RGBX, the fourth byte ignored), rows `stride` bytes
// apart. 1 on success, 0 when a buffer or the file could not be had.
int png_write_rgb(const char *path, const uint8_t *px, uint32_t w, uint32_t h, uint32_t step, size_t stride,
                  PngBufs *b);
void png_bufs_free(PngBufs *b);
