// Adapted from MacNativeGames games/totk/runtime/motion_fusion.c @ b75be8cf (MIT, Copyright (c) 2026 MacNativeGames
// contributors).
// Six-axis sensor states (nn::hid::SixAxisSensorState) from gyro and accelerometer samples. Input per sample, in the
// Switch controller frame (x to the right, y toward the top edge, z out of the face): acceleration in G including
// gravity, as gravity reads (lying face up z = -1 G), and rotation rate in rad/s. Output per sample, as the Switch
// reports it: acceleration in G, angular velocity in revolutions per second, angle = the integrated revolutions per
// axis, direction = the controller's three axes in a reference frame whose z is up. The orientation is integrated from
// the gyro and, while the controller is not being accelerated (|a| within 15 % of 1 G), pulled toward the measured
// gravity (Mahony, gain 0.5/s: about 2 s), as the Switch's sensor fusion does; yaw has no reference and drifts with the
// gyro's bias. The gyro's zero-rate bias (a resting reading that is not 0, a few degrees a second on some pads, the
// Steam Controller's among them) is measured while the controller is still and taken off every sample, as the Switch
// does with its factory and stillness calibration: uncorrected, a resting controller reads as a slow steady turn, and a
// gyro cursor (blade mode, the bow) creeps across the screen. The newest MOTION_RING samples are kept; reads are newest first. A restart (StartSixAxisSensor) drops
// the samples before it: reads return nothing until the next sample, which starts a fresh frame and angle; sampling
// numbers go on. The feeding thread writes, guest threads read, both under the instance's SRW lock.
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "host.h"
#include "input.h"
#include "platform/platform.h"

enum { MOTION_RING = 64 };
#define STALE_NS 200000000ull   // no sample for 200 ms: no motion
// Bias calibration: a stretch of STILL_NS in which every sample's rate is under STILL_RATE, the rates stay within
// STILL_SPREAD of each other per axis and the acceleration within 10 % of 1 G is a controller at rest (on a desk, or
// held quite still); its mean rate, when under BIAS_MAX (a slow deliberate turn is not taken for bias), is the bias,
// blended in by half after the first one.
#define STILL_NS 500000000ull
#define STILL_RATE 0.15     // rad/s, about 8.6 deg/s
#define STILL_SPREAD 0.06   // rad/s
#define BIAS_MAX 0.07       // rad/s, about 4 deg/s

struct MotionFusion {
    PlatMutex lock;
    MotionState ring[MOTION_RING];
    uint64_t produced, last_ns;
    uint64_t first;   // produced at the last restart: older samples are not read
    double R[3][3];   // controller -> reference; row i = reference axis i in controller coordinates
    double angle[3];
    int restart;
    double bias[3];                         // gyro zero-rate bias, rad/s, taken off every sample
    int calibrated;
    int turn;                               // motion_set_turn
    double still_sum[3], still_lo[3], still_hi[3];
    uint64_t still_from, still_n;           // still_from: time of the stretch's first sample, 0 none
};

MotionFusion *motion_new(void) {
    MotionFusion *m = calloc(1, sizeof *m);
    if (!m) rt_fatal("motion: out of memory");
    plat_mutex_init(&m->lock);
    m->restart = 1;
    return m;
}

static MotionFusion *global;

static void global_init(void *arg) {
    (void)arg;
    global = motion_new();
}

MotionFusion *motion_global(void) {
    static PlatOnce once = PLAT_ONCE_INIT;
    plat_once(&once, global_init, NULL);
    return global;
}

static double norm3(const double v[3]) { return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
static void cross3(const double a[3], const double b[3], double out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}
// Rows, from the up row (the one the gravity correction keeps right).
static void orthonormalize(double R[3][3]) {
    double n = norm3(R[2]);
    for (int j = 0; j < 3; j++) R[2][j] /= n;
    double d = R[0][0] * R[2][0] + R[0][1] * R[2][1] + R[0][2] * R[2][2];
    for (int j = 0; j < 3; j++) R[0][j] -= d * R[2][j];
    n = norm3(R[0]);
    for (int j = 0; j < 3; j++) R[0][j] /= n;
    cross3(R[2], R[0], R[1]);
}
// A fresh orientation: z up against the measured gravity, x as close to the controller's x as that allows.
static void align_to(double R[3][3], const double acc[3]) {
    double n = norm3(acc), up[3] = {0, 0, 1}, x[3] = {1, 0, 0};
    if (n > 0.5)
        for (int j = 0; j < 3; j++) up[j] = -acc[j] / n;
    if (fabs(up[0]) > 0.9) { x[0] = 0; x[1] = 1; }
    memcpy(R[2], up, sizeof up);
    memcpy(R[0], x, sizeof x);
    orthonormalize(R);
}
// Body-frame rate: R = R * exp([w] dt), the rate plus the gravity pull.
static void integrate(double R[3][3], const double gyro[3], const double acc[3], double dt) {
    double w[3] = {gyro[0], gyro[1], gyro[2]}, n = norm3(acc);
    if (fabs(n - 1) < 0.15) {   // not being accelerated: turn the estimated down toward the measured one
        const double kp = 0.5;
        double measured[3] = {acc[0] / n, acc[1] / n, acc[2] / n}, estimated[3] = {-R[2][0], -R[2][1], -R[2][2]}, e[3];
        cross3(measured, estimated, e);
        for (int j = 0; j < 3; j++) w[j] += kp * e[j];
    }
    double rate = norm3(w), theta = rate * dt;
    if (theta < 1e-12) return;
    double k[3] = {w[0] / rate, w[1] / rate, w[2] / rate}, s = sin(theta), c = 1 - cos(theta);
    double K[3][3] = {{0, -k[2], k[1]}, {k[2], 0, -k[0]}, {-k[1], k[0], 0}}, E[3][3], out[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double kk = 0;
            for (int q = 0; q < 3; q++) kk += K[i][q] * K[q][j];
            E[i][j] = (i == j) + s * K[i][j] + c * kk;
        }
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double v = 0;
            for (int q = 0; q < 3; q++) v += R[i][q] * E[q][j];
            out[i][j] = v;
        }
    memcpy(R, out, sizeof out);
    orthonormalize(R);
}

// Under the lock: one raw sample into the still stretch; at its end the bias is updated.
static void calibrate(MotionFusion *m, const double acc[3], const double gyro[3], uint64_t now) {
    double n = norm3(acc);
    int still = fabs(n - 1) < 0.1 && norm3(gyro) < STILL_RATE;
    for (int i = 0; still && m->still_n && i < 3; i++)
        still = fmax(m->still_hi[i], gyro[i]) - fmin(m->still_lo[i], gyro[i]) < STILL_SPREAD;
    if (!still) { m->still_from = m->still_n = 0; return; }
    if (!m->still_n) {
        m->still_from = now;
        for (int i = 0; i < 3; i++) m->still_sum[i] = 0, m->still_lo[i] = m->still_hi[i] = gyro[i];
    }
    for (int i = 0; i < 3; i++) {
        m->still_sum[i] += gyro[i];
        m->still_lo[i] = fmin(m->still_lo[i], gyro[i]);
        m->still_hi[i] = fmax(m->still_hi[i], gyro[i]);
    }
    m->still_n++;
    if (now - m->still_from < STILL_NS) return;
    double mean[3] = {m->still_sum[0] / m->still_n, m->still_sum[1] / m->still_n, m->still_sum[2] / m->still_n};
    m->still_from = m->still_n = 0;
    if (norm3(mean) >= BIAS_MAX) return;
    for (int i = 0; i < 3; i++) m->bias[i] = m->calibrated ? (m->bias[i] + mean[i]) / 2 : mean[i];
    if (!m->calibrated && m == global)
        rt_log("motion: gyro bias %.2f %.2f %.2f deg/s, taken off every sample", m->bias[0] * 180 / M_PI,
               m->bias[1] * 180 / M_PI, m->bias[2] * 180 / M_PI);
    m->calibrated = 1;
}

void motion_feed(MotionFusion *m, const double acc[3], const double raw[3], uint64_t now) {
    plat_mutex_lock(&m->lock);
    calibrate(m, acc, raw, now);
    const double gyro[3] = {raw[0] - m->bias[0], raw[1] - m->bias[1], raw[2] - m->bias[2]};
    double dt = m->last_ns && now > m->last_ns ? (double)(now - m->last_ns) / 1e9 : 0.004;
    if (dt > 0.05) dt = 0.004;   // a gap (no controller, sensors were off): no integration across it
    m->last_ns = now;
    if (m->restart) {
        align_to(m->R, acc);
        memset(m->angle, 0, sizeof m->angle);
        m->restart = 0;
    } else integrate(m->R, gyro, acc, dt);
    MotionState *s = &m->ring[m->produced % MOTION_RING];
    *s = (MotionState){.delta_ns = (int64_t)(dt * 1e9), .sampling = (int64_t)m->produced + 1, .attributes = 1};
    for (int i = 0; i < 3; i++) {
        double rps = gyro[i] / (2 * M_PI);
        // angle y also takes the turn about z, with z's sign: the game's aim reads -y as right, as tested in blade mode
        m->angle[i] += (i == 1 ? rps + m->turn * gyro[2] / (2 * M_PI) : rps) * dt;
        s->acc[i] = (float)acc[i];
        s->gyro[i] = (float)rps;
        s->angle[i] = (float)m->angle[i];
        for (int j = 0; j < 3; j++) s->dir[i * 3 + j] = (float)m->R[j][i];   // direction i = controller axis i in the reference frame
    }
    m->produced++;
    plat_mutex_unlock(&m->lock);
}

void motion_set_turn(MotionFusion *m, int k) {
    plat_mutex_lock(&m->lock);
    m->turn = k;
    plat_mutex_unlock(&m->lock);
}

int motion_read(MotionFusion *m, MotionState *out, int max, uint64_t now) {
    if (max <= 0) return 0;
    int n = 0;
    plat_mutex_lock_shared(&m->lock);
    if (m->produced > m->first && (now < m->last_ns || now - m->last_ns < STALE_NS)) {
        uint64_t have = m->produced - m->first < MOTION_RING ? m->produced - m->first : MOTION_RING;
        n = (int)(have < (uint64_t)max ? have : (uint64_t)max);
        for (int i = 0; i < n; i++) out[i] = m->ring[(m->produced - 1 - (uint64_t)i) % MOTION_RING];
    }
    plat_mutex_unlock_shared(&m->lock);
    return n;
}

void motion_restart(MotionFusion *m) {
    plat_mutex_lock(&m->lock);
    m->restart = 1;
    m->first = m->produced;
    m->last_ns = 0;   // the first sample after the restart integrates nothing across the stop
    memset(m->angle, 0, sizeof m->angle);
    plat_mutex_unlock(&m->lock);
}
