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

#if 1
/* opt-07: mont_mul / mont_mul_ui32 / mont_reduce_wide are straight-line
   collective code: one instance per TPI segment (TPI<=32, per-warp packing)
   and every warp lane participating (callers must not exit lanes mid-op).
   The compile-time full-warp mask lets ptxas inline each SHFL instead of
   emitting the ~7-slot __cuda_sm70_shflsync_idx_p helper call that a runtime
   instance mask produces.  Instance-divergent callers (gcd, modinv, the
   conditional -n fix inside mont_reduce_wide) must keep sync_mask(). */
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

template<class env>
__device__ __forceinline__ void core_t<env>::mont_mul(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS], const uint32_t n[LIMBS], const uint32_t np0) {
  uint32_t sync=CGBN_MONT_SYNC, group_thread=threadIdx.x & TPI-1;
  uint32_t t, t0, t1, q, r1, ra[LIMBS+2], ru[LIMBS+1], c=0;
    
  #pragma unroll
  for(int32_t index=0;index<=LIMBS;index++) {
    ra[index]=0;
    ru[index]=0;
  }
  ra[LIMBS+1]=0;
  
  #pragma unroll
  for(int32_t thread=0;thread<TPI;thread+=2) {
    #pragma unroll
    for(int32_t word=0;word<2*LIMBS;word+=2) {
      if(word<LIMBS) 
        t0=__shfl_sync(sync, b[word], thread, TPI);
      else
        t0=__shfl_sync(sync, b[word-LIMBS], thread+1, TPI);
      if(word+1<LIMBS)
        t1=__shfl_sync(sync, b[word+1], thread, TPI);
      else
        t1=__shfl_sync(sync, b[word+1-LIMBS], thread+1, TPI);
    
      /* FIRST HALF */
      
      chain_t<> chain1;                               // unaligned: T0 * A_odd
      ra[0]=chain1.add(ra[0], c);
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index+=2) {
        ru[index]=chain1.madlo(a[index+1], t0, ru[index]);
        ru[index+1]=chain1.madhi(a[index+1], t0, ru[index+1]);
      }
      if(LIMBS%2==1)
        ru[LIMBS-1]=chain1.add(ru[LIMBS-1], 0);
        
      chain_t<> chain2;                               // aligned:   T0 * A_even
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index+=2) {
        ra[index]=chain2.madlo(a[index], t0, ra[index]);
        ra[index+1]=chain2.madhi(a[index], t0, ra[index+1]);
      }
      if(LIMBS%2==0)
        ra[LIMBS]=chain2.add(ra[LIMBS], 0);
      
      chain_t<> chain3;                               // aligned:   Q0 * N_even
      q=__shfl_sync(sync, ra[0], 0, TPI)*np0;
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index+=2) {
        ra[index]=chain3.madlo(n[index], q, ra[index]);
        ra[index+1]=chain3.madhi(n[index], q, ra[index+1]);
      }
      ra[LIMBS+LIMBS%2]=chain3.add(ra[LIMBS+LIMBS%2], 0);
              
      chain_t<> chain4;                               // unaligned: Q0 * N_odd
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index+=2) {
        ru[index]=chain4.madlo(n[index+1], q, ru[index]);
        ru[index+1]=chain4.madhi(n[index+1], q, ru[index+1]);
      }
      ru[LIMBS-LIMBS%2]=chain4.add(ru[LIMBS-LIMBS%2], 0);

      /* SECOND HALF */

      t0=ra[0];

      chain_t<> chain5;                               // unaigned: T1 * A_even
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index+=2) {
        ru[index]=chain5.madlo(a[index], t1, ru[index]);
        ru[index+1]=chain5.madhi(a[index], t1, ru[index+1]);
      }
      ru[LIMBS-LIMBS%2]=chain5.add(ru[LIMBS-LIMBS%2], 0);
      
      chain_t<> chain6;                               // aligned:   T1 * A_odd
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index+=2) {
        ra[index+2]=chain6.madlo(a[index+1], t1, ra[index+2]);
        ra[index+3]=chain6.madhi(a[index+1], t1, ra[index+3]);
      }
      if(LIMBS%2==1)
        ra[LIMBS+1]=chain3.add(ra[LIMBS+1], 0);

      chain_t<> chain7;                               // aligned:   Q1 * N_odd
      ru[0]=chain7.add(ru[0], ra[1]);
      q=__shfl_sync(sync, ru[0], 0, TPI)*np0;
      #pragma unroll
      for(int32_t index=0;index<(int32_t)LIMBS-3;index+=2) {
        ra[index]=chain7.madlo(n[index+1], q, ra[index+2]);
        ra[index+1]=chain7.madhi(n[index+1], q, ra[index+3]);
      }
      ra[LIMBS-2-LIMBS%2]=chain3.madlo(n[LIMBS-1-LIMBS%2], q, ra[LIMBS-LIMBS%2]);
      ra[LIMBS-1-LIMBS%2]=chain3.madhi(n[LIMBS-1-LIMBS%2], q, ra[LIMBS+1-LIMBS%2]);
      if(LIMBS%2==1) {
        ra[LIMBS-1]=chain3.add(ra[LIMBS+1], 0);
        ra[LIMBS]=0;
      }
      else 
        ra[LIMBS-LIMBS%2]=chain3.add(0, 0);

      chain_t<> chain8;                               // unaigned: Q1 * N_even
      t1=chain8.madlo(n[0], q, ru[0]);
      c=chain8.madhi(n[0], q, ru[1]);
      #pragma unroll
      for(int32_t index=0;index<LIMBS-2;index+=2) {
        ru[index]=chain8.madlo(n[index+2], q, ru[index+2]);
        ru[index+1]=chain8.madhi(n[index+2], q, ru[index+3]);
      }
      if(LIMBS%2==1)
        ru[LIMBS-1]=chain8.add(0, 0);
      else {
        ru[LIMBS-2]=chain8.add(ru[LIMBS], 0);
        ru[LIMBS-1]=0;
      }
      ru[LIMBS]=0;
            
      t0=__shfl_sync(sync, t0, threadIdx.x+1, TPI);
      t1=__shfl_sync(sync, t1, threadIdx.x+1, TPI);
        
      ra[LIMBS-2]=add_cc(ra[LIMBS-2], t0);
      ra[LIMBS-1]=addc_cc(ra[LIMBS-1], t1);
      ra[LIMBS]=addc(ra[LIMBS], 0);
      ra[LIMBS+1]=0;
    }
  }
  
  chain_t<LIMBS+1> chain9;
  r[0]=chain9.add(ra[0], c);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++) 
    r[index]=chain9.add(ra[index], ru[index-1]);
  r1=chain9.add(ra[LIMBS], ru[LIMBS-1]);

  // r1:r0 <= 0x00000002 0xFFFFFFFD
  t=__shfl_up_sync(sync, r1, 1, TPI);
  
  // all but most significant thread clears r1
  if(group_thread!=TPI-1)
    r1=0;
  if(group_thread==0)
    t=0;

  chain_t<LIMBS+1> chain10;
  r[0]=chain10.add(r[0], t);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    r[index]=chain10.add(r[index], 0);
  c=chain10.add(r1, 0);

  if(PADDING==0) c=-fast_propagate_add_conv(c, r); else c=-fast_propagate_add(c, r);

  // compute -n
  t=n[0]-(group_thread==0);   // n must be odd, so there is no chance for a carry ripple

  chain_t<LIMBS+1> chain11;
  r[0]=chain11.add(r[0], ~t & c);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    r[index]=chain11.add(r[index], ~n[index] & c);
  c=chain11.add(0, 0);
  if(PADDING==0) fast_propagate_add_conv(c, r); else fast_propagate_add(c, r);
  clear_padding(r);
}
#endif

/* r = a * d / 2^32 mod n -- one Montgomery limb step with a 32-bit scalar.
   a in [0, 2^(BITS-1)), d in [0, 2^32), n odd and n >= 2^32; returns r < 2n.
   The wide product w = a*d (L+1 significant words) is formed locally with the
   boundary carry folded into the next thread, placed into the canonical wide
   layout shifted up by L-1 limbs (w * 2^(32*(L-1))), and reduced with one
   mont_reduce_wide -- which then yields w / 2^32 mod n. */
template<class env>
__device__ __forceinline__ void core_t<env>::mont_mul_ui32(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t d, const uint32_t n[LIMBS], const uint32_t np0) {
  uint32_t sync=CGBN_MONT_SYNC, group_thread=threadIdx.x & TPI-1;
  uint32_t lo[LIMBS], hi[LIMBS], c, carry;

  // w = a*d: lo[index] = word group_thread*LIMBS+index; the running high word
  // crosses into the next thread's lo[0]
  c=0;
  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++) {
    lo[index]=madlo_cc(a[index], d, c);   // lo + carry from the previous word
    c=madhic(a[index], d, 0);             // hi(a*d) + the madlo carry; c itself
  }                                      // was already consumed by madlo_cc

  carry=__shfl_up_sync(sync, c, 1, TPI);
  if(group_thread==0)
    carry=0;
  {
    uint32_t s=lo[0]+carry, cy=(s<lo[0]);
    lo[0]=s;
    #pragma unroll
    for(int32_t index=1;index<LIMBS;index++) {
      s=lo[index]+cy; cy=(s<lo[index]); lo[index]=s;
    }
    c=c+cy;   // top thread: word TPI*LIMBS of w (multi-bit; keep exact)
  }

  // shift w up by L-1 limbs inside the canonical wide layout:
  //   w' word t*LIMBS+i   (t*LIMBS+i < L)  = 0 except w' word L-1 = w word 0
  //   w' word L+t*LIMBS+i              = w word t*LIMBS+i+1
  {
    uint32_t next_lo0=__shfl_down_sync(sync, lo[0], 1, TPI);  // top thread: own value (unused)
    uint32_t w0=__shfl_sync(sync, lo[0], 0, TPI);

    #pragma unroll
    for(int32_t index=0;index<LIMBS-1;index++)
      hi[index]=lo[index+1];
    hi[LIMBS-1]=(group_thread==TPI-1) ? c : next_lo0;

    #pragma unroll
    for(int32_t index=0;index<LIMBS;index++)
      lo[index]=0;
    if(group_thread==TPI-1)
      lo[LIMBS-1]=w0;
  }

  mont_reduce_wide(r, lo, hi, n, np0, false);
  clear_padding(r);
}

template<class env>
__device__ __forceinline__ void core_t<env>::mont_reduce_wide(uint32_t r[LIMBS], const uint32_t lo[LIMBS], const uint32_t hi[LIMBS], const uint32_t n[LIMBS], const uint32_t np0, const bool zero) {
  uint32_t sync=CGBN_MONT_SYNC, group_thread=threadIdx.x & TPI-1;
  uint32_t t0, t1, q, ra[LIMBS+2], ru[LIMBS+1], c=0, top;

  #pragma unroll
  for(int32_t index=0;index<LIMBS;index++) {
    ra[index]=lo[index];
    ru[index]=0;
  }
  ra[LIMBS]=0;
  ru[LIMBS]=0;
  ra[LIMBS+1]=0;
  
  c=0;
  #pragma nounroll
  for(int32_t thread=0;thread<TPI;thread+=2) {
    #pragma unroll
    for(int32_t l=0;l<2*LIMBS;l+=2) {
      chain_t<> chain1;                               // unaligned: Q0 * N_odd
      ra[0]=chain1.add(ra[0], c);
      q=__shfl_sync(sync, ra[0], 0, TPI)*np0;
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index+=2) {
        ru[index]=chain1.madlo(n[index+1], q, ru[index]);
        ru[index+1]=chain1.madhi(n[index+1], q, ru[index+1]);
      }
      if(LIMBS%2==1)
        ru[LIMBS-1]=chain1.add(0, 0);
 
      chain_t<> chain2;                               // aligned:   Q0 * N_even
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index+=2) {
        ra[index]=chain2.madlo(n[index], q, ra[index]);
        ra[index+1]=chain2.madhi(n[index], q, ra[index+1]);
      }
      if(LIMBS%2==0)
        ra[LIMBS]=chain2.add(ra[LIMBS], 0);      

      t0=__shfl_sync(sync, ra[0], threadIdx.x+1, TPI);
      if(!zero) {
        if(l<LIMBS)
          top=__shfl_sync(sync, hi[l], thread, TPI);
        else
          top=__shfl_sync(sync, hi[l-LIMBS], thread+1, TPI);
        t0=(group_thread==TPI-1) ? top : t0;
      }        

      chain_t<> chain3;                               // aligned:   Q1 * N_odd
      ru[0]=chain3.add(ru[0], ra[1]);
      q=__shfl_sync(sync, ru[0], 0, TPI)*np0;
      #pragma unroll
      for(int32_t index=0;index<(int32_t)LIMBS-3;index+=2) {
        ra[index]=chain3.madlo(n[index+1], q, ra[index+2]);
        ra[index+1]=chain3.madhi(n[index+1], q, ra[index+3]);
      }
      ra[LIMBS-2-LIMBS%2]=chain3.madlo(n[LIMBS-1-LIMBS%2], q, ra[LIMBS-LIMBS%2]);
      ra[LIMBS-1-LIMBS%2]=chain3.madhi(n[LIMBS-1-LIMBS%2], q, ra[LIMBS+1-LIMBS%2]);
      if(LIMBS%2==1)
        ra[LIMBS-1]=chain3.add(0, 0);

      chain_t<> chain4;                               // unaligned: Q1 * N_even
      t1=chain4.madlo(n[0], q, ru[0]);
      c=chain4.madhi(n[0], q, ru[1]);
      #pragma unroll
      for(int32_t index=0;index<LIMBS-2;index+=2) {
        ru[index]=chain4.madlo(n[index+2], q, ru[index+2]);
        ru[index+1]=chain4.madhi(n[index+2], q, ru[index+3]);
      }
      if(LIMBS%2==1)
        ru[LIMBS-1]=0;
      else 
        ru[LIMBS-2]=chain4.add(0, 0);
      ru[LIMBS-1+LIMBS%2]=0;

      t1=__shfl_sync(sync, t1, threadIdx.x+1, TPI);
      if(!zero) {
        if(l+1<LIMBS)
          top=__shfl_sync(sync, hi[l+1], thread, TPI);
        else
          top=__shfl_sync(sync, hi[l+1-LIMBS], thread+1, TPI);
        t1=(group_thread==TPI-1) ? top : t1;
      }
      
      ra[LIMBS-2]=add_cc(ra[LIMBS-2], t0);
      ra[LIMBS-1]=addc_cc(ra[LIMBS-1], t1);
      ra[LIMBS]=addc(0, 0);
    }
  }
  
  chain_t<> chain5;
  ra[0]=chain5.add(ra[0], c);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    ra[index]=chain5.add(ra[index], ru[index-1]);
  c=chain5.add(ra[LIMBS], 0);
  
  if(PADDING==0) c=fast_propagate_add_conv(c, ra); else c=fast_propagate_add(c, ra);
  
  if(!zero && c!=0) {
    t0=n[0]-(group_thread==0);   // n must be odd, so there is no chance for a carry ripple

    chain_t<LIMBS+1> chain3;
    ra[0]=chain3.add(ra[0], ~t0);
    #pragma unroll
    for(int32_t index=1;index<LIMBS;index++)
      ra[index]=chain3.add(ra[index], ~n[index]);
    c=chain3.add(0, 0);
    fast_propagate_add(c, ra);
    clear_padding(ra);
  }

  mpset<LIMBS>(r, ra);
}

#if 0
template<uint32_t LIMBS>
__device__ __forceinline__ void fwmont_mul(uint32_t r[LIMBS], const uint32_t a[LIMBS], const uint32_t b[LIMBS], const uint32_t n[LIMBS], const uint32_t np0) {
  uint32_t sync=sync_mask(), group_thread=threadIdx.x & TPI-1;
  uint32_t t, t0, t1, q, r1, ra[LIMBS+2], ru[LIMBS+1], c=0;
    
  #pragma unroll
  for(int32_t index=0;index<=LIMBS;index++) {
    ra[index]=0;
    ru[index]=0;
  }
  ra[LIMBS+1]=0;
  
  #pragma nounroll
  for(int32_t thread=0;thread<TPI;thread+=2) {
    #pragma unroll
    for(int32_t word=0;word<2*LIMBS;word+=2) {
      if(word<LIMBS) 
        t0=__shfl_sync(sync, b[word], thread, TPI);
      else
        t0=__shfl_sync(sync, b[word-LIMBS], thread+1, TPI);
      if(word+1<LIMBS)
        t1=__shfl_sync(sync, b[word+1], thread, TPI);
      else
        t1=__shfl_sync(sync, b[word+1-LIMBS], thread+1, TPI);
    
      /* FIRST HALF */
      
      chain_t<> chain1;                               // aligned:   T0 * A_even
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index+=2) {
        ra[index]=chain1.madlo(a[index], t0, ra[index]);
        ra[index+1]=chain1.madhi(a[index], t0, ra[index+1]);
      }
      ra[LIMBS+LIMBS%2]=chain1.add(ra[LIMBS+LIMBS%2], 0);

      chain_t<> chain2;                               // unaligned: T0 * A_odd
      ra[0]=chain2.add(ra[0], c);
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index+=2) {
        ru[index]=chain2.madlo(a[index+1], t0, ru[index]);
        ru[index+1]=chain2.madhi(a[index+1], t0, ru[index+1]);
      }
      ru[LIMBS-LIMBS%2]=chain2.add(ru[LIMBS-LIMBS%2], 0);
        
      q=__shfl_sync(sync, ra[0], 0, TPI)*np0;

      chain_t<> chain3;                               // aligned:   Q0 * N_even
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index+=2) {
        ra[index]=chain3.madlo(n[index], q, ra[index]);
        ra[index+1]=chain3.madhi(n[index], q, ra[index+1]);
      }
      t=__shfl_sync(sync, ra[0], threadIdx.x+1, TPI);
      if(LIMBS%2==0)
        ra[LIMBS]=chain3.add(ra[LIMBS], t);
      ra[LIMBS+1]=chain3.add(ra[LIMBS+1], 0);
        
      chain_t<> chain4;                               // unaligned: Q0 * N_odd
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index+=2) {
        ru[index]=chain4.madlo(n[index+1], q, ru[index]);
        ru[index+1]=chain4.madhi(n[index+1], q, ru[index+1]);
      }
      if(LIMBS%2==1) 
        ru[LIMBS-1]=chain4.add(ru[LIMBS-1], t);
      ru[LIMBS]=chain4.add(ru[LIMBS], 0);
            
      /* SECOND HALF */

      chain_t<> chain5;                               // unaigned: T1 * A_even
      #pragma unroll
      for(int32_t index=0;index<LIMBS;index+=2) {
        ru[index]=chain5.madlo(a[index], t1, ru[index]);
        ru[index+1]=chain5.madhi(a[index], t1, ru[index+1]);
      }
      if(LIMBS%2==0)
        ru[LIMBS]=chain5.add(ru[LIMBS], 0);
      
      chain_t<> chain6;                               // aligned:   T1 * A_odd
      ru[0]=chain6.add(ru[0], ra[1]);
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index+=2) {
        ra[index+2]=chain6.madlo(a[index+1], t1, ra[index+2]);
        ra[index+3]=chain6.madhi(a[index+1], t1, ra[index+3]);
      }
      if(LIMBS%2==1)
        ra[LIMBS+1]=chain6.add(ra[LIMBS+1], 0);
      
      q=__shfl_sync(sync, ru[0], 0, TPI)*np0;

      chain_t<> chain7;                               // unaigned: Q1 * N_even
      t=chain7.madlo(n[0], q, ru[0]);
      c=chain7.madhi(n[0], q, ru[1]);
      #pragma unroll
      for(int32_t index=0;index<LIMBS-2;index+=2) {
        ru[index]=chain7.madlo(n[index+2], q, ru[index+2]);
        ru[index+1]=chain7.madhi(n[index+2], q, ru[index+3]);
      }
      t=__shfl_sync(sync, t, threadIdx.x+1, TPI);
      if(LIMBS%2==0)
        ru[LIMBS-2]=chain7.add(ru[LIMBS], t);
      ru[LIMBS-1]=chain7.add(0, 0);
      ru[LIMBS]=0;
            
      chain_t<> chain8;                               // aligned:   Q1 * N_odd
      #pragma unroll
      for(int32_t index=0;index<LIMBS-1;index+=2) {
        ra[index]=chain8.madlo(n[index+1], q, ra[index+2]);
        ra[index+1]=chain8.madhi(n[index+1], q, ra[index+3]);
      }
      if(LIMBS%2==1)
        ra[LIMBS-1]=chain8.add(ra[LIMBS+1], t);
      ra[LIMBS]=chain8.add(0, 0);
      ra[LIMBS+1]=0;
    }
  }
  
  chain_t<LIMBS+1> chain9;
  r[0]=chain9.add(ra[0], c);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++) 
    r[index]=chain9.add(ra[index], ru[index-1]);
  r1=chain9.add(ra[LIMBS], ru[LIMBS-1]);

  // r1:r0 <= 0x00000002 0xFFFFFFFD
  t=__shfl_up_sync(sync, r1, 1, TPI);
  
  // all but most significant thread clears r1
  if(group_thread!=TPI-1)
    r1=0;
  if(group_thread==0)
    t=0;

  chain_t<LIMBS+1> chain10;
  r[0]=chain10.add(r[0], t);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    r[index]=chain10.add(r[index], 0);
  c=chain10.add(r1, 0);

  c=-fast_propagate_add(c, r);

  // compute -n
  t=n[0]-(group_thread==0);   // n must be odd, so there is no chance for a carry ripple

  chain_t<LIMBS+1> chain11;
  r[0]=chain11.add(r[0], ~t & c);
  #pragma unroll
  for(int32_t index=1;index<LIMBS;index++)
    r[index]=chain11.add(r[index], ~n[index] & c);
  c=chain11.add(0, 0);
  fast_propagate_add(c, r);
  clear_padding(r);
}
#endif

} /* namespace cgbn */
