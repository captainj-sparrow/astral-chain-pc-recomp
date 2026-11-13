#version 450
// Ambient occlusion, last step (runtime/vk_ao.c run), drawn with hdr_fullscreen.vert into the game's light target with
// a multiplying blend (the target's colour times this output). The factor on the light comes from ao_blur.comp at the
// occlusion's resolution, the target's or half of it, one per colour channel (the game's fog keeps its light, and it
// is coloured): occlusion texel k holds target pixel k * scale (ao_prep.comp), and its blurred value sits half a texel
// before it on each axis (ao_blur.comp). Each pixel blends the four texels around it by their distance and by how
// close their view z is to its own, so occlusion stays on its side of a depth edge; a pixel none of them matches (a
// thin edge between two surfaces) takes the one nearest in depth.
// Sky pixels (cleared depth) and pixels left open are discarded: the blend then writes nothing there.
// debug: the blend is off and every pixel shows the factor in place of the light.
layout(binding = 0) uniform sampler2D ao_tex;      // .rgb the factor on the light, .a view z (0 sky)
layout(binding = 1) uniform sampler2D depth_tex;   // the scene depth, the target's size
layout(location = 0) out vec4 frag;
layout(push_constant) uniform P {
    vec4 inv2, inv3;   // rows 2 and 3 of the inverse projection, row-major
    vec4 size;         // target width, height, 1 for -1..1 clip depth (0 for 0..1), 0
    vec4 mode;         // x: 1 debug view; y: 0; z: scale, 1 or 2
} pc;

void main() {
    ivec2 q = ivec2(gl_FragCoord.xy);
    float d = texelFetch(depth_tex, q, 0).r;
    vec3 k = vec3(1.0);
    if (d < 1.0) {
        vec4 ndc = vec4(gl_FragCoord.x / pc.size.x * 2.0 - 1.0, 1.0 - gl_FragCoord.y / pc.size.y * 2.0,
                        pc.size.z > 0.5 ? d * 2.0 - 1.0 : d, 1.0);
        float z = dot(pc.inv2, ndc) / dot(pc.inv3, ndc);
        int s = int(pc.mode.z);
        vec2 u = vec2(q) / float(s) + 0.5;   // in texels, the blur's half texel taken back
        ivec2 i0 = ivec2(u), hs = textureSize(ao_tex, 0) - 1;
        vec2 f = fract(u);
        float tol = 0.04 * abs(z) + 0.05, wsum = 0.0, near = 1e30;
        vec3 sum = vec3(0.0), nearv = vec3(1.0);
        for (int y = 0; y < 2; y++)
            for (int x = 0; x < 2; x++) {
                vec4 a = texelFetch(ao_tex, min(i0 + ivec2(x, y), hs), 0);
                if (a.a == 0.0) continue;
                float dz = abs(a.a - z);
                float w = (x == 0 ? 1.0 - f.x : f.x) * (y == 0 ? 1.0 - f.y : f.y) * clamp(2.0 - 2.0 * dz / tol, 0.0, 1.0);
                sum += a.rgb * w;
                wsum += w;
                if (dz < near) near = dz, nearv = a.rgb;
            }
        k = clamp(wsum > 1e-3 ? sum / wsum : nearv, 0.0, 1.0);
    }
    if (pc.mode.x < 0.5 && min(k.r, min(k.g, k.b)) > 0.996) discard;
    frag = vec4(k, 1.0);
}
