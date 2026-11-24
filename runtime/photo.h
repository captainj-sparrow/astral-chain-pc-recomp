// Photo mode capture (photo.c): the picture behind each photo the game takes, saved as a PNG and handed to the game's
// album in place of its own capture. nvn_api_cmd.c records the capture, vk_photo.c makes the copy on the GPU.
#pragma once
#include <stddef.h>
#include <stdint.h>

#define PHOTO_ALBUM_W 1280   // the album's pictures, as the game makes them (main+0x57b150 reads them back)
#define PHOTO_ALBUM_H 720

// nvn_api_cmd.c: the game's photo function on guest thread `thread` asked for a photo whose passes draw into the
// NVNtexture `album` (stage 1, rt_photo_arm; album NULL when unknown) or has recorded those passes (stage 3,
// rt_photo_end).
void nvn_photo_stage(const void *thread, int stage, const void *album);
// The NVN layer records a photo's capture: its number, and the shape of the frame into *shape (resolution.c
// res_wide_shape, 0 for the picture's own).
uint32_t photo_take(float *shape);
// The rectangle {x0, y0, x1, y1} of a w x h output that the middle of the picture as shown (the output stretched to
// the frame's shape, 0: its own) at shape `want` comes from.
void photo_middle(uint32_t w, uint32_t h, float shape, double want, double r[4]);
// The back end could not copy photo seq; the game keeps its own capture.
void photo_capture_failed(uint32_t seq, const char *why);
// On the completion thread: the copied output of photo seq, w x h pixels of 4 bytes, rows stride bytes apart, BGRA
// when bgra. px stays valid until photo.c calls release(ctx), from another thread.
void photo_captured(uint32_t seq, const uint8_t *px, uint32_t w, uint32_t h, size_t stride, int bgra, float shape,
                    void (*release)(void *), void *ctx);
