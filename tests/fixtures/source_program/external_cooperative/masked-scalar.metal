#include <metal_stdlib>
using namespace metal;
// CPU prototype. Host must enforce the bound full-tile admission. Not device qualified.
#pragma clang fp contract(off) reassociate(off)
kernel void _fwd(device float* b0 [[buffer(0)]],
device float* b1 [[buffer(1)]],
device const float* b2 [[buffer(2)]],
device const float* b3 [[buffer(3)]],
device const float* b4 [[buffer(4)]],
device const float* b5 [[buffer(5)]],
device const uchar* b6 [[buffer(6)]],
constant float& a7 [[buffer(7)]],
constant float& a8 [[buffer(8)]],
constant int& a9 [[buffer(9)]],
constant int& a10 [[buffer(10)]],
uint3 pid [[threadgroup_position_in_grid]],
uint row [[thread_index_in_threadgroup]]) {
if (row >= 32u) return;
float m = as_type<float>(0xff800000u); float l = 0.0f; float acc[32];
for (uint c=0u; c<32u; ++c) acc[c]=0.0f;
float t0;
t0 = a7 * as_type<float>(0x3fb8aa3bu);
float t1[32];
for (uint c = 0u; c < 32u; ++c) {
t1[c] = b2[((ulong)((long)as_type<int>((pid.z * ((as_type<uint>(a9) * as_type<uint>(a9)) * 32u)))) + (ulong)((long)as_type<int>((pid.y * (as_type<uint>(a9) * 32u)))) + (ulong)((long)as_type<int>(((row + (pid.x * 32u)) * 32u))) + (ulong)((long)as_type<int>(c)))] * t0;
}
for (uint iv=0u; iv<as_type<uint>(a9); iv+=32u) {
float t2[32];
for (uint c = 0u; c < 32u; ++c) {
t2[c] = b5[((ulong)((long)as_type<int>((pid.z * (as_type<uint>(a9) * as_type<uint>(a9))))) + (ulong)((long)as_type<int>(((row + (pid.x * 32u)) * as_type<uint>(a9)))) + (ulong)((long)as_type<int>((c + iv))))] * as_type<float>(0x3fb8aa3bu);
}
float t3[32];
for (uint c = 0u; c < 32u; ++c) {
t3[c] = t2[c];
for (uint k = 0u; k < 32u; ++k) {
float product = t1[k] * b3[((ulong)((long)as_type<int>((pid.z * ((as_type<uint>(a9) * as_type<uint>(a9)) * 32u)))) + (ulong)((long)as_type<int>((pid.y * (as_type<uint>(a9) * 32u)))) + (ulong)((long)as_type<int>(k)) + (ulong)((long)as_type<int>(((c + iv) * 32u))))];
t3[c] = t3[c] + product;
}
}
float t4[32];
for (uint c = 0u; c < 32u; ++c) {
t4[c] = ((b6[((ulong)((long)as_type<int>(((as_type<uint>(as_type<int>(pid.z) / as_type<int>(as_type<uint>(a10)))) * (as_type<uint>(a9) * as_type<uint>(a9))))) + (ulong)((long)as_type<int>((pid.y * as_type<uint>(a9)))) + (ulong)((long)as_type<int>((c + iv))))] != uchar(0)) ? a8 : t3[c]);
}
float t5[32];
for (uint c = 0u; c < 32u; ++c) {
t5[c] = (((as_type<int>((row + (pid.x * 32u))) < as_type<int>(as_type<uint>(a9))) && (as_type<int>((c + iv)) < as_type<int>(as_type<uint>(a9)))) ? t4[c] : a8);
}
float t6;
t6 = t5[0u];
for (uint k = 1u; k < 32u; ++k) {
t6 = fmax(t6, t5[k]);
}
float t7;
t7 = fmax(m, t6);
float t8;
t8 = m - t7;
float t9;
t9 = precise::exp2(t8);
float t10;
t10 = l * t9;
float t11[32];
for (uint c = 0u; c < 32u; ++c) {
t11[c] = t5[c] - t7;
}
float t12[32];
for (uint c = 0u; c < 32u; ++c) {
t12[c] = precise::exp2(t11[c]);
}
float t13;
t13 = t12[0u];
for (uint k = 1u; k < 32u; ++k) {
t13 = t13 + t12[k];
}
float t14;
t14 = t10 + t13;
float t15[32];
for (uint c = 0u; c < 32u; ++c) {
t15[c] = acc[c] * t9;
}
float t16[32];
for (uint c = 0u; c < 32u; ++c) {
t16[c] = t15[c];
for (uint k = 0u; k < 32u; ++k) {
float product = t12[k] * b4[((ulong)((long)as_type<int>((pid.z * ((as_type<uint>(a9) * as_type<uint>(a9)) * 32u)))) + (ulong)((long)as_type<int>((pid.y * (as_type<uint>(a9) * 32u)))) + (ulong)((long)as_type<int>(((k + iv) * 32u))) + (ulong)((long)as_type<int>(c)))];
t16[c] = t16[c] + product;
}
}
m = t7;
l = t14;
for (uint c=0u; c<32u; ++c) acc[c] = t16[c];
}
float t17[32];
for (uint c = 0u; c < 32u; ++c) {
t17[c] = acc[c] / l;
}
float t18;
t18 = m * as_type<float>(0x3f317218u);
float t19;
t19 = precise::log(l);
float t20;
t20 = t18 + t19;
if (as_type<int>(row + pid.x*32u) < a9) {
for (uint c=0u; c<32u; ++c) b0[((ulong)((long)as_type<int>((pid.z * ((as_type<uint>(a9) * as_type<uint>(a9)) * 32u)))) + (ulong)((long)as_type<int>((pid.y * (as_type<uint>(a9) * 32u)))) + (ulong)((long)as_type<int>(((row + (pid.x * 32u)) * 32u))) + (ulong)((long)as_type<int>(c)))] = t17[c];
b1[((ulong)((long)as_type<int>((pid.z * (as_type<uint>(a9) * as_type<uint>(a9))))) + (ulong)((long)as_type<int>((pid.y * as_type<uint>(a9)))) + (ulong)((long)as_type<int>((row + (pid.x * 32u)))))] = t20;
}
}
