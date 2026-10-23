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

/* opt-07: straight-line collective code (see core_mont_wmad.cu) -- the
   compile-time full-warp shfl mask avoids the __cuda_sm70_shflsync helper
   call that a runtime instance mask produces on TPI<32. */
/* opt-07: the inline-SHFL codegen is scoped to PADDING==0 shapes.  On the
 * padded path (e.g. 544/8) constant-folding the shfl/ballot masks to
 * 0xFFFFFFFF triggers an nvcc 13.0 miscompile of the same family as the
 * documented cgbn_sub full-container anomaly (dirty words above BITS/32 in
 * one-iteration mulcheck runs; shape-isolated to padded mont codegen, green
 * on every PADDING==0 shape incl. odd-LIMBS 768/8 and 384/4).  The ternary
 * folds at compile time: padded instantiations keep the base instance-mask
 * path bit-for-bit, unpadded ones get the inlined collectives. */
#define CGBN_MONT_SYNC ((PADDING==0)?CGBN_WARP_SYNC:sync_mask())
#define CGBN_WARP_SYNC 0xFFFFFFFFu

/* single limb version */
template<class env>
__device__ __forceinline__ void core_t<env>::mont_mul(uint32_t &r, const uint32_t a, const uint32_t b, const uint32_t n, const uint32_t np0) {
  uint32_t sync=CGBN_MONT_SYNC, group_thread=threadIdx.x & TPI-1;
  uint32_t r0=0, r1=0, r2, t, c;
  
  #pragma unroll
  for(int32_t thread=0;thread<TPI;thread++) {
    // broadcast b[i]
    t=__shfl_sync(sync, b, thread, TPI);

    r0=madlo_cc(a, t, r0);
    r1=madhic_cc(a, t, r1);
    r2=addc(0, 0);

    // broadcast r[0]
    t=__shfl_sync(sync, r0, 0, TPI)*np0;
    r0=madlo_cc(n, t, r0);
    r1=madhic_cc(n, t, r1);
    r2=addc(r2, 0);
    
    // shift right by 32 bits (top thread gets zero)
    r0=__shfl_sync(sync, r0, threadIdx.x+1, TPI);
    r0=add_cc(r0, r1);
    r1=addc(r2, 0);
  }
 
  // r1:r0 <= 0x00000002 0xFFFFFFFD
  t=__shfl_up_sync(sync, r1, 1, TPI);
  
  // all but most significant thread clears r1
  if(group_thread!=TPI-1)
    r1=0;
  if(group_thread==0)
    t=0;

  r0=add_cc(r0, t);
  c=addc(r1, 0);

  if(PADDING==0) c=fast_propagate_add_conv(c, r0); else c=fast_propagate_add(c, r0);
  
  // compute -n
  t=n-(group_thread==0);   // n must be odd, so there is no chance for a carry ripple
  t=~t & -c;

  r0=add_cc(r0, t);
  c=addc(0, 0);
  if(PADDING==0) fast_propagate_add_conv(c, r0); else fast_propagate_add(c, r0);

  r=r0; 
}

/* r = a * d / 2^32 mod n -- one Montgomery limb step with a 32-bit scalar.
   a in [0, 2^BITS), d in [0, 2^32), n odd and n >= 2^32; returns r < 2n.
   Costs one product row plus one REDC row instead of the TPI rows of
   mont_mul(a, d*R^{TPI-1}), i.e. the fused form of ecm's special_mult_ui32. */
template<class env>
__device__ __forceinline__ void core_t<env>::mont_mul_ui32(uint32_t &r, const uint32_t a, const uint32_t d, const uint32_t n, const uint32_t np0) {
  uint32_t sync=CGBN_MONT_SYNC, group_thread=threadIdx.x & TPI-1;
  uint32_t r0, r1, r2, t, c;

  // w = a*d; this thread holds word group_thread (r0) and word group_thread+1 (r1)
  r0=madlo(a, d, 0);
  r1=madhi(a, d, 0);

  // single REDC step: q = w_low * np0 zeroes lane 0, so the top-lane wrap is benign
  t=__shfl_sync(sync, r0, 0, TPI)*np0;
  r0=madlo_cc(n, t, r0);
  r1=madhic_cc(n, t, r1);
  r2=addc(0, 0);

  // shift right by 32 bits (top thread gets the zeroed lane 0)
  r0=__shfl_sync(sync, r0, threadIdx.x+1, TPI);
  r0=add_cc(r0, r1);
  r1=addc(r2, 0);

  // same tail as mont_mul
  t=__shfl_up_sync(sync, r1, 1, TPI);

  if(group_thread!=TPI-1)
    r1=0;
  if(group_thread==0)
    t=0;

  r0=add_cc(r0, t);
  c=addc(r1, 0);

  if(PADDING==0) c=fast_propagate_add_conv(c, r0); else c=fast_propagate_add(c, r0);

  // compute -n
  t=n-(group_thread==0);   // n must be odd, so there is no chance for a carry ripple
  t=~t & -c;

  r0=add_cc(r0, t);
  c=addc(0, 0);
  if(PADDING==0) fast_propagate_add_conv(c, r0); else fast_propagate_add(c, r0);

  r=r0;
}

template<class env>
__device__ __forceinline__ void core_t<env>::mont_reduce_wide(uint32_t &r, const uint32_t lo, const uint32_t hi, const uint32_t n, const uint32_t np0, const bool zero) {
  uint32_t sync=CGBN_MONT_SYNC, group_thread=threadIdx.x & TPI-1;
  uint32_t r0=lo, r1=0, t, top;
  
  #pragma unroll
  for(int32_t thread=0;thread<TPI;thread++) {
    t=__shfl_sync(sync, r0, 0, TPI)*np0;
    r0=madlo_cc(n, t, r0);
    r1=madhic_cc(n, t, r1);
        
    // shift right by 32 bits (top thread gets zero)
    r0=__shfl_sync(sync, r0, threadIdx.x+1, TPI);
    if(!zero) {
      top=__shfl_sync(sync, hi, thread, TPI);
      r0=(group_thread==TPI-1) ? top : r0;
    }
    
    // add it in
    r0=add_cc(r0, r1);
    r1=addc(0, 0);
  }
 
  if(PADDING==0) r1=fast_propagate_add_conv(r1, r0); else r1=fast_propagate_add(r1, r0);

  if(!zero && r1!=0) {
    // compute -n
    t=n-(group_thread==0);   // n must be odd, so there is no chance for a carry ripple
    t=~t & -r1;

    r0=add_cc(r0, t);
    r1=addc(0, 0);
    fast_propagate_add(r1, r0);
  }
  r=r0; 
}

} /* namespace cgbn */

