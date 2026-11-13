#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off) reassociate(off)
// Exact 128-thread full-tile lowering. External caller must enforce metadata.
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
uint tid [[thread_index_in_threadgroup]]) {
threadgroup float Q[1056];
threadgroup float S[1056];
threadgroup float KV[1056];
threadgroup float M[32];
threadgroup float L[32];
threadgroup float A[32];
threadgroup uchar Mask[32];
// EXECUTOR_BEGIN
float acc[8];
float scale = a7 * as_type<float>(0x3fb8aa3bu);
for (uint slot = 0u; slot < 8u; ++slot) {
uint linear = tid + 128u * slot;
uint r = linear / 32u;
uint c = linear % 32u;
acc[slot] = 0.0f;
Q[r * 33u + c] = b2[((ulong)((long)as_type<int>((pid.z * ((as_type<uint>(a9) * as_type<uint>(a9)) * 32u)))) + (ulong)((long)as_type<int>((pid.y * (as_type<uint>(a9) * 32u)))) + (ulong)((long)as_type<int>(((r + (pid.x * 32u)) * 32u))) + (ulong)((long)as_type<int>(c)))] * scale;
}
if (tid < 32u) {
M[tid] = as_type<float>(0xff800000u);
L[tid] = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint iv = 0u; iv < as_type<uint>(a9); iv += 32u) {
for (uint slot = 0u; slot < 8u; ++slot) {
uint linear = tid + 128u * slot;
uint r = linear / 32u;
uint c = linear % 32u;
KV[r * 33u + c] = b3[((ulong)((long)as_type<int>((pid.z * ((as_type<uint>(a9) * as_type<uint>(a9)) * 32u)))) + (ulong)((long)as_type<int>((pid.y * (as_type<uint>(a9) * 32u)))) + (ulong)((long)as_type<int>(r)) + (ulong)((long)as_type<int>(((c + iv) * 32u))))];
S[r * 33u + c] = b5[((ulong)((long)as_type<int>((pid.z * (as_type<uint>(a9) * as_type<uint>(a9))))) + (ulong)((long)as_type<int>(((r + (pid.x * 32u)) * as_type<uint>(a9)))) + (ulong)((long)as_type<int>((c + iv))))] * as_type<float>(0x3fb8aa3bu);
}
if (tid < 32u) {
Mask[tid] = b6[((ulong)((long)as_type<int>(((as_type<uint>(as_type<int>(pid.z) / as_type<int>(as_type<uint>(a10)))) * (as_type<uint>(a9) * as_type<uint>(a9))))) + (ulong)((long)as_type<int>((pid.y * as_type<uint>(a9)))) + (ulong)((long)as_type<int>((tid + iv))))];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint slot = 0u; slot < 8u; ++slot) {
uint linear = tid + 128u * slot;
uint r = linear / 32u;
uint c = linear % 32u;
float score;
score = S[r * 33u + c];
for (uint k = 0u; k < 32u; ++k) {
float product = Q[r * 33u + k] * KV[k * 33u + c];
score = score + product;
}
score = (Mask[c] != uchar(0) ? a8 : score);
score = ((as_type<int>(r + pid.x * 32u) < a9 && as_type<int>(c + iv) < a9) ? score : a8);
S[r * 33u + c] = score;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid < 32u) {
float maximum = S[tid * 33u];
for (uint k = 1u; k < 32u; ++k) {
maximum = fmax(maximum, S[tid * 33u + k]);
}
maximum = fmax(M[tid], maximum);
float difference = M[tid] - maximum;
A[tid] = precise::exp2(difference);
M[tid] = maximum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint slot = 0u; slot < 8u; ++slot) {
uint linear = tid + 128u * slot;
uint r = linear / 32u;
uint c = linear % 32u;
float difference = S[r * 33u + c] - M[r];
S[r * 33u + c] = precise::exp2(difference);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid < 32u) {
float total = S[tid * 33u];
for (uint k = 1u; k < 32u; ++k) {
total = total + S[tid * 33u + k];
}
float rescaled = L[tid] * A[tid];
L[tid] = rescaled + total;
}
for (uint slot = 0u; slot < 8u; ++slot) {
uint linear = tid + 128u * slot;
uint r = linear / 32u;
uint c = linear % 32u;
KV[r * 33u + c] = b4[((ulong)((long)as_type<int>((pid.z * ((as_type<uint>(a9) * as_type<uint>(a9)) * 32u)))) + (ulong)((long)as_type<int>((pid.y * (as_type<uint>(a9) * 32u)))) + (ulong)((long)as_type<int>(((r + iv) * 32u))) + (ulong)((long)as_type<int>(c)))];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint slot = 0u; slot < 8u; ++slot) {
uint linear = tid + 128u * slot;
uint r = linear / 32u;
uint c = linear % 32u;
float carry = acc[slot] * A[r];
acc[slot] = carry;
for (uint k = 0u; k < 32u; ++k) {
float product = S[r * 33u + k] * KV[k * 33u + c];
acc[slot] = acc[slot] + product;
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint slot = 0u; slot < 8u; ++slot) {
uint linear = tid + 128u * slot;
uint r = linear / 32u;
uint c = linear % 32u;
if (as_type<int>(r + pid.x * 32u) < a9) {
b0[((ulong)((long)as_type<int>((pid.z * ((as_type<uint>(a9) * as_type<uint>(a9)) * 32u)))) + (ulong)((long)as_type<int>((pid.y * (as_type<uint>(a9) * 32u)))) + (ulong)((long)as_type<int>(((r + (pid.x * 32u)) * 32u))) + (ulong)((long)as_type<int>(c)))] = acc[slot] / L[r];
}
}
if (tid < 32u) {
if (as_type<int>(tid + pid.x * 32u) < a9) {
float scaled_maximum = M[tid] * as_type<float>(0x3f317218u);
float log_l = precise::log(L[tid]);
b1[((ulong)((long)as_type<int>((pid.z * (as_type<uint>(a9) * as_type<uint>(a9))))) + (ulong)((long)as_type<int>((pid.y * as_type<uint>(a9)))) + (ulong)((long)as_type<int>((tid + (pid.x * 32u)))))] = scaled_maximum + log_l;
}
}
}
