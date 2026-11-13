#version 450
// Ambient occlusion, last step (runtime/vk_ao.c run), drawn with hdr_fullscreen.vert into the game's light target with
// a multiplying blend (the target's colour times this output). Half-resolution texel k holds full-resolution pixel 2k
// (ao_prep.comp); each pixel averages the 4x4 texels around its own, the span of ao.comp's 4x4 pattern, so the pattern
// evens out, and weighs each by how close its view z is to this pixel's, so occlusion does not bleed across depth
// edges. The weight is 1 over the near half of the tolerance: on a surface that slants away the 16 texels then count
// alike, where a weight falling from the first centimetre left the pattern showing as a grid of dots on walls.
// The averaged visibility becomes the factor on the light through a fixed curve (vk_ao.c AO_OPEN, AO_POWER):
// visibilities from "open" up are 1, the rest, scaled to 0..1, is raised to "power".
// Sky pixels (cleared depth) and pixels left open are discarded: the blend then writes nothing there.
// debug: the blend is off and every pixel shows the AO factor in grey.
layout(binding = 0) uniform sampler2D ao_tex;      // ao.comp's output: .x visibility, .y view z (0 sky)
layout(binding = 1) uniform sampler2D depth_tex;   // the scene depth, full resolution
layout(location = 0) out vec4 frag;
layout(push_constant) uniform P {
    vec4 inv2, inv3;   // rows 2 and 3 of the inverse projection, row-major
    vec4 size;         // full width, full height, 1 for -1..1 clip depth (0 for 0..1), power
    vec4 mode;         // x: 1 debug view; y: open, the visibility from which nothing darkens
} pc;

void main() {
    ivec2 q = ivec2(gl_FragCoord.xy);
    float d = texelFetch(depth_tex, q, 0).r;
    float k = 1.0;
    if (d < 1.0) {
        vec4 ndc = vec4(gl_FragCoord.x / pc.size.x * 2.0 - 1.0, 1.0 - gl_FragCoord.y / pc.size.y * 2.0,
                        pc.size.z > 0.5 ? d * 2.0 - 1.0 : d, 1.0);
        float z = dot(pc.inv2, ndc) / dot(pc.inv3, ndc);
        ivec2 hp = q >> 1, hs = textureSize(ao_tex, 0) - 1;
        float tol = 0.04 * abs(z) + 0.05, sum = 0.0, wsum = 0.0;
        for (int y = -2; y < 2; y++)
            for (int x = -2; x < 2; x++) {
                vec2 a = texelFetch(ao_tex, clamp(hp + ivec2(x, y), ivec2(0), hs), 0).xy;
                float w = a.y == 0.0 ? 0.0 : clamp(2.0 - 2.0 * abs(a.y - z) / tol, 0.0, 1.0);
                sum += a.x * w;
                wsum += w;
            }
        k = pow(clamp((wsum > 1e-3 ? sum / wsum : 1.0) / pc.mode.y, 0.0, 1.0), pc.size.w);
    }
    if (pc.mode.x < 0.5 && k > 0.996) discard;
    frag = vec4(k, k, k, 1.0);
}
