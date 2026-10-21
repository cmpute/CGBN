/***

Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

***/

namespace cgbn {

template<class env> 
__device__ __forceinline__ int32_t core_t<env>::add(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS]) {
  uint32_t carry;

  chain_t<> chain;
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++)
    r[index]=chain.add(a[index], b[index]);
  carry=chain.add(0, 0);
  return fast_propagate_add(carry, r);
}

template<class env> 
__device__ __forceinline__ int32_t core_t<env>::sub(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS]) {
  uint32_t carry;
  
  chain_t<> chain;
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++)
    r[index]=chain.sub(a[index], b[index]);
  carry=chain.sub(0, 0);
  return -fast_propagate_sub(carry, r);
} 

/* OPTIM item 04: fused modular add/subtract.  r = (a +/- b) mod m with the
 * correction folded into a single API call; requires a, b in [0, m).  The
 * two propagate passes are the same as the manual sub/compare/add pattern --
 * the value of the API is a single well-defined primitive whose select logic
 * the compiler can schedule against the surrounding ladder (and a future
 * single-pass combined resolver has a home). */
template<class env>
__device__ __forceinline__ void core_t<env>::mod_sub(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS], const uint32_t m[LIMBS]) {
  uint32_t carry;
  
  chain_t<> chain;
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++)
    r[index]=chain.sub(a[index], b[index]);
  carry=chain.sub(0, 0);
  carry=-fast_propagate_sub(carry, r);

  if(carry!=0) {
    chain_t<> chain2;
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      r[index]=chain2.add(r[index], m[index]);
    carry=chain2.add(0, 0);
    fast_propagate_add(carry, r);
  }
}

template<class env>
__device__ __forceinline__ void core_t<env>::mod_add(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS], const uint32_t m[LIMBS]) {
  uint32_t carry;
  
  chain_t<> chain;
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++)
    r[index]=chain.add(a[index], b[index]);
  carry=chain.add(0, 0);
  fast_propagate_add(carry, r);

  if(core_t<env>::compare(r, m)>=0) {
    chain_t<> chain2;
    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      r[index]=chain2.sub(r[index], m[index]);
    carry=chain2.sub(0, 0);
    fast_propagate_sub(carry, r);
  }
}

template<class env> 
__device__ __forceinline__ int32_t core_t<env>::negate(uint32_t r[LIMBS], const uint32_t a[LIMBS]) {
  mpset<LIMBS>(r, a);
  return fast_negate(r);
} 

} /* namespace cgbn */
