#include "Ray.h"

#include <immintrin.h>

////////////////////////// Use Julien Pommier's library as not every compiler has _mm_cos_ps and the like available

/* SIMD (SSE1+MMX or SSE2) implementation of sin, cos, exp and log

   Inspired by Intel Approximate Math library, and based on the
   corresponding algorithms of the cephes math library

   The default is to use the SSE1 version. If you define USE_SSE2 the
   the SSE2 intrinsics will be used in place of the MMX intrinsics. Do
   not expect any significant performance improvement with SSE2.
*/

/* Copyright (C) 2007  Julien Pommier

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
	 claim that you wrote the original software. If you use this software
	 in a product, an acknowledgment in the product documentation would be
	 appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
	 misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

/* yes I know, the top of this file is quite ugly */

#ifndef USE_SSE2
#define USE_SSE2
#endif

#ifdef _MSC_VER /* visual c++ */
#define ALIGN16_BEG __declspec(align(16))
#define ALIGN16_END
#else /* gcc or icc */
#define ALIGN16_BEG
#define ALIGN16_END __attribute__((aligned(16)))
#endif

/* __m128 is ugly to write */
typedef __m128 v4sf; // vector of 4 float (sse1)

#ifdef USE_SSE2
#include <emmintrin.h>
typedef __m128i v4si; // vector of 4 int (sse2)
#else
typedef __m64 v2si; // vector of 2 int (mmx)
#endif

/* declare some SSE constants -- why can't I figure a better way to do that? */
#define _PS_CONST(Name, Val) static const ALIGN16_BEG float _ps_##Name[4] ALIGN16_END = {Val, Val, Val, Val}
#define _PI32_CONST(Name, Val) static const ALIGN16_BEG int _pi32_##Name[4] ALIGN16_END = {Val, Val, Val, Val}
#define _PS_CONST_TYPE(Name, Type, Val) static const ALIGN16_BEG Type _ps_##Name[4] ALIGN16_END = {Val, Val, Val, Val}

_PS_CONST(1, 1.0f);
_PS_CONST(0p5, 0.5f);
/* the smallest non denormalized float number */
_PS_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS_CONST_TYPE(sign_mask, int, (int)0x80000000);
_PS_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST(1, 1);
_PI32_CONST(inv1, ~1);
_PI32_CONST(2, 2);
_PI32_CONST(4, 4);
_PI32_CONST(0x7f, 0x7f);

_PS_CONST(cephes_SQRTHF, 0.707106781186547524);
_PS_CONST(cephes_log_p0, 7.0376836292E-2);
_PS_CONST(cephes_log_p1, -1.1514610310E-1);
_PS_CONST(cephes_log_p2, 1.1676998740E-1);
_PS_CONST(cephes_log_p3, -1.2420140846E-1);
_PS_CONST(cephes_log_p4, +1.4249322787E-1);
_PS_CONST(cephes_log_p5, -1.6668057665E-1);
_PS_CONST(cephes_log_p6, +2.0000714765E-1);
_PS_CONST(cephes_log_p7, -2.4999993993E-1);
_PS_CONST(cephes_log_p8, +3.3333331174E-1);
_PS_CONST(cephes_log_q1, -2.12194440e-4);
_PS_CONST(cephes_log_q2, 0.693359375);

#ifndef USE_SSE2
typedef union xmm_mm_union {
	__m128 xmm;
	__m64 mm[2];
} xmm_mm_union;

#define COPY_XMM_TO_MM(xmm_, mm0_, mm1_)                                                                                                                       \
	{                                                                                                                                                          \
		xmm_mm_union u;                                                                                                                                        \
		u.xmm = xmm_;                                                                                                                                          \
		mm0_ = u.mm[0];                                                                                                                                        \
		mm1_ = u.mm[1];                                                                                                                                        \
	}

#define COPY_MM_TO_XMM(mm0_, mm1_, xmm_)                                                                                                                       \
	{                                                                                                                                                          \
		xmm_mm_union u;                                                                                                                                        \
		u.mm[0] = mm0_;                                                                                                                                        \
		u.mm[1] = mm1_;                                                                                                                                        \
		xmm_ = u.xmm;                                                                                                                                          \
	}

#endif // USE_SSE2

/* natural logarithm computed for 4 simultaneous float
   return NaN for x <= 0
*/
v4sf log_ps(v4sf x)
{
#ifdef USE_SSE2
	v4si emm0;
#else
	v2si mm0, mm1;
#endif
	v4sf one = *(v4sf *)_ps_1;

	v4sf invalid_mask = _mm_cmple_ps(x, _mm_setzero_ps());

	x = _mm_max_ps(x, *(v4sf *)_ps_min_norm_pos); /* cut off denormalized stuff */

#ifndef USE_SSE2
	/* part 1: x = frexpf(x, &e); */
	COPY_XMM_TO_MM(x, mm0, mm1);
	mm0 = _mm_srli_pi32(mm0, 23);
	mm1 = _mm_srli_pi32(mm1, 23);
#else
	emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);
#endif
	/* keep only the fractional part */
	x = _mm_and_ps(x, *(v4sf *)_ps_inv_mant_mask);
	x = _mm_or_ps(x, *(v4sf *)_ps_0p5);

#ifndef USE_SSE2
	/* now e=mm0:mm1 contain the really base-2 exponent */
	mm0 = _mm_sub_pi32(mm0, *(v2si *)_pi32_0x7f);
	mm1 = _mm_sub_pi32(mm1, *(v2si *)_pi32_0x7f);
	v4sf e = _mm_cvtpi32x2_ps(mm0, mm1);
	_mm_empty(); /* bye bye mmx */
#else
	emm0 = _mm_sub_epi32(emm0, *(v4si *)_pi32_0x7f);
	v4sf e = _mm_cvtepi32_ps(emm0);
#endif

	e = _mm_add_ps(e, one);

	/* part2:
	   if( x < SQRTHF ) {
		 e -= 1;
		 x = x + x - 1.0;
	   } else { x = x - 1.0; }
	*/
	v4sf mask = _mm_cmplt_ps(x, *(v4sf *)_ps_cephes_SQRTHF);
	v4sf tmp = _mm_and_ps(x, mask);
	x = _mm_sub_ps(x, one);
	e = _mm_sub_ps(e, _mm_and_ps(one, mask));
	x = _mm_add_ps(x, tmp);

	v4sf z = _mm_mul_ps(x, x);

	v4sf y = *(v4sf *)_ps_cephes_log_p0;
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf *)_ps_cephes_log_p1);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf *)_ps_cephes_log_p2);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf *)_ps_cephes_log_p3);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf *)_ps_cephes_log_p4);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf *)_ps_cephes_log_p5);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf *)_ps_cephes_log_p6);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf *)_ps_cephes_log_p7);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf *)_ps_cephes_log_p8);
	y = _mm_mul_ps(y, x);

	y = _mm_mul_ps(y, z);

	tmp = _mm_mul_ps(e, *(v4sf *)_ps_cephes_log_q1);
	y = _mm_add_ps(y, tmp);

	tmp = _mm_mul_ps(z, *(v4sf *)_ps_0p5);
	y = _mm_sub_ps(y, tmp);

	tmp = _mm_mul_ps(e, *(v4sf *)_ps_cephes_log_q2);
	x = _mm_add_ps(x, y);
	x = _mm_add_ps(x, tmp);
	x = _mm_or_ps(x, invalid_mask); // negative arg will be NAN
	return x;
}

_PS_CONST(exp_hi, 88.3762626647949f);
_PS_CONST(exp_lo, -88.3762626647949f);

_PS_CONST(cephes_LOG2EF, 1.44269504088896341);
_PS_CONST(cephes_exp_C1, 0.693359375);
_PS_CONST(cephes_exp_C2, -2.12194440e-4);

_PS_CONST(cephes_exp_p0, 1.9875691500E-4);
_PS_CONST(cephes_exp_p1, 1.3981999507E-3);
_PS_CONST(cephes_exp_p2, 8.3334519073E-3);
_PS_CONST(cephes_exp_p3, 4.1665795894E-2);
_PS_CONST(cephes_exp_p4, 1.6666665459E-1);
_PS_CONST(cephes_exp_p5, 5.0000001201E-1);

v4sf exp_ps(v4sf x)
{
	v4sf tmp = _mm_setzero_ps(), fx;
#ifdef USE_SSE2
	v4si emm0;
#else
	v2si mm0, mm1;
#endif
	v4sf one = *(v4sf *)_ps_1;

	x = _mm_min_ps(x, *(v4sf *)_ps_exp_hi);
	x = _mm_max_ps(x, *(v4sf *)_ps_exp_lo);

	/* express exp(x) as exp(g + n*log(2)) */
	fx = _mm_mul_ps(x, *(v4sf *)_ps_cephes_LOG2EF);
	fx = _mm_add_ps(fx, *(v4sf *)_ps_0p5);

	/* how to perform a floorf with SSE: just below */
#ifndef USE_SSE2
	/* step 1 : cast to int */
	tmp = _mm_movehl_ps(tmp, fx);
	mm0 = _mm_cvttps_pi32(fx);
	mm1 = _mm_cvttps_pi32(tmp);
	/* step 2 : cast back to float */
	tmp = _mm_cvtpi32x2_ps(mm0, mm1);
#else
	emm0 = _mm_cvttps_epi32(fx);
	tmp = _mm_cvtepi32_ps(emm0);
#endif
	/* if greater, substract 1 */
	v4sf mask = _mm_cmpgt_ps(tmp, fx);
	mask = _mm_and_ps(mask, one);
	fx = _mm_sub_ps(tmp, mask);

	tmp = _mm_mul_ps(fx, *(v4sf *)_ps_cephes_exp_C1);
	v4sf z = _mm_mul_ps(fx, *(v4sf *)_ps_cephes_exp_C2);
	x = _mm_sub_ps(x, tmp);
	x = _mm_sub_ps(x, z);

	z = _mm_mul_ps(x, x);

	v4sf y = *(v4sf *)_ps_cephes_exp_p0;
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf *)_ps_cephes_exp_p1);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf *)_ps_cephes_exp_p2);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf *)_ps_cephes_exp_p3);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf *)_ps_cephes_exp_p4);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf *)_ps_cephes_exp_p5);
	y = _mm_mul_ps(y, z);
	y = _mm_add_ps(y, x);
	y = _mm_add_ps(y, one);

	/* build 2^n */
#ifndef USE_SSE2
	z = _mm_movehl_ps(z, fx);
	mm0 = _mm_cvttps_pi32(fx);
	mm1 = _mm_cvttps_pi32(z);
	mm0 = _mm_add_pi32(mm0, *(v2si *)_pi32_0x7f);
	mm1 = _mm_add_pi32(mm1, *(v2si *)_pi32_0x7f);
	mm0 = _mm_slli_pi32(mm0, 23);
	mm1 = _mm_slli_pi32(mm1, 23);

	v4sf pow2n;
	COPY_MM_TO_XMM(mm0, mm1, pow2n);
	_mm_empty();
#else
	emm0 = _mm_cvttps_epi32(fx);
	emm0 = _mm_add_epi32(emm0, *(v4si *)_pi32_0x7f);
	emm0 = _mm_slli_epi32(emm0, 23);
	v4sf pow2n = _mm_castsi128_ps(emm0);
#endif
	y = _mm_mul_ps(y, pow2n);
	return y;
}

_PS_CONST(minus_cephes_DP1, -0.78515625);
_PS_CONST(minus_cephes_DP2, -2.4187564849853515625e-4);
_PS_CONST(minus_cephes_DP3, -3.77489497744594108e-8);
_PS_CONST(sincof_p0, -1.9515295891E-4);
_PS_CONST(sincof_p1, 8.3321608736E-3);
_PS_CONST(sincof_p2, -1.6666654611E-1);
_PS_CONST(coscof_p0, 2.443315711809948E-005);
_PS_CONST(coscof_p1, -1.388731625493765E-003);
_PS_CONST(coscof_p2, 4.166664568298827E-002);
_PS_CONST(cephes_FOPI, 1.27323954473516); // 4 / M_PI

/* evaluation of 4 sines at onces, using only SSE1+MMX intrinsics so
   it runs also on old athlons XPs and the pentium III of your grand
   mother.

   The code is the exact rewriting of the cephes sinf function.
   Precision is excellent as long as x < 8192 (I did not bother to
   take into account the special handling they have for greater values
   -- it does not return garbage for arguments over 8192, though, but
   the extra precision is missing).

   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
   surprising but correct result.

   Performance is also surprisingly good, 1.33 times faster than the
   macos vsinf SSE2 function, and 1.5 times faster than the
   __vrs4_sinf of amd's ACML (which is only available in 64 bits). Not
   too bad for an SSE1 function (with no special tuning) !
   However the latter libraries probably have a much better handling of NaN,
   Inf, denormalized and other special arguments..

   On my core 1 duo, the execution of this function takes approximately 95 cycles.

   From what I have observed on the experiments with Intel AMath lib, switching to an
   SSE2 version would improve the perf by only 10%.

   Since it is based on SSE intrinsics, it has to be compiled at -O2 to
   deliver full speed.
*/
v4sf sin_ps(v4sf x)
{ // any x
	v4sf xmm1, xmm2 = _mm_setzero_ps(), xmm3, sign_bit, y;

#ifdef USE_SSE2
	v4si emm0, emm2;
#else
	v2si mm0, mm1, mm2, mm3;
#endif
	sign_bit = x;
	/* take the absolute value */
	x = _mm_and_ps(x, *(v4sf *)_ps_inv_sign_mask);
	/* extract the sign bit (upper one) */
	sign_bit = _mm_and_ps(sign_bit, *(v4sf *)_ps_sign_mask);

	/* scale by 4/Pi */
	y = _mm_mul_ps(x, *(v4sf *)_ps_cephes_FOPI);

#ifdef USE_SSE2
	/* store the integer part of y in mm0 */
	emm2 = _mm_cvttps_epi32(y);
	/* j=(j+1) & (~1) (see the cephes sources) */
	emm2 = _mm_add_epi32(emm2, *(v4si *)_pi32_1);
	emm2 = _mm_and_si128(emm2, *(v4si *)_pi32_inv1);
	y = _mm_cvtepi32_ps(emm2);

	/* get the swap sign flag */
	emm0 = _mm_and_si128(emm2, *(v4si *)_pi32_4);
	emm0 = _mm_slli_epi32(emm0, 29);
	/* get the polynom selection mask
	   there is one polynom for 0 <= x <= Pi/4
	   and another one for Pi/4<x<=Pi/2

	   Both branches will be computed.
	*/
	emm2 = _mm_and_si128(emm2, *(v4si *)_pi32_2);
	emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());

	v4sf swap_sign_bit = _mm_castsi128_ps(emm0);
	v4sf poly_mask = _mm_castsi128_ps(emm2);
	sign_bit = _mm_xor_ps(sign_bit, swap_sign_bit);

#else
	/* store the integer part of y in mm0:mm1 */
	xmm2 = _mm_movehl_ps(xmm2, y);
	mm2 = _mm_cvttps_pi32(y);
	mm3 = _mm_cvttps_pi32(xmm2);
	/* j=(j+1) & (~1) (see the cephes sources) */
	mm2 = _mm_add_pi32(mm2, *(v2si *)_pi32_1);
	mm3 = _mm_add_pi32(mm3, *(v2si *)_pi32_1);
	mm2 = _mm_and_si64(mm2, *(v2si *)_pi32_inv1);
	mm3 = _mm_and_si64(mm3, *(v2si *)_pi32_inv1);
	y = _mm_cvtpi32x2_ps(mm2, mm3);
	/* get the swap sign flag */
	mm0 = _mm_and_si64(mm2, *(v2si *)_pi32_4);
	mm1 = _mm_and_si64(mm3, *(v2si *)_pi32_4);
	mm0 = _mm_slli_pi32(mm0, 29);
	mm1 = _mm_slli_pi32(mm1, 29);
	/* get the polynom selection mask */
	mm2 = _mm_and_si64(mm2, *(v2si *)_pi32_2);
	mm3 = _mm_and_si64(mm3, *(v2si *)_pi32_2);
	mm2 = _mm_cmpeq_pi32(mm2, _mm_setzero_si64());
	mm3 = _mm_cmpeq_pi32(mm3, _mm_setzero_si64());
	v4sf swap_sign_bit, poly_mask;
	COPY_MM_TO_XMM(mm0, mm1, swap_sign_bit);
	COPY_MM_TO_XMM(mm2, mm3, poly_mask);
	sign_bit = _mm_xor_ps(sign_bit, swap_sign_bit);
	_mm_empty(); /* good-bye mmx */
#endif

	/* The magic pass: "Extended precision modular arithmetic"
	   x = ((x - y * DP1) - y * DP2) - y * DP3; */
	xmm1 = *(v4sf *)_ps_minus_cephes_DP1;
	xmm2 = *(v4sf *)_ps_minus_cephes_DP2;
	xmm3 = *(v4sf *)_ps_minus_cephes_DP3;
	xmm1 = _mm_mul_ps(y, xmm1);
	xmm2 = _mm_mul_ps(y, xmm2);
	xmm3 = _mm_mul_ps(y, xmm3);
	x = _mm_add_ps(x, xmm1);
	x = _mm_add_ps(x, xmm2);
	x = _mm_add_ps(x, xmm3);

	/* Evaluate the first polynom  (0 <= x <= Pi/4) */
	y = *(v4sf *)_ps_coscof_p0;
	v4sf z = _mm_mul_ps(x, x);

	y = _mm_mul_ps(y, z);
	y = _mm_add_ps(y, *(v4sf *)_ps_coscof_p1);
	y = _mm_mul_ps(y, z);
	y = _mm_add_ps(y, *(v4sf *)_ps_coscof_p2);
	y = _mm_mul_ps(y, z);
	y = _mm_mul_ps(y, z);
	v4sf tmp = _mm_mul_ps(z, *(v4sf *)_ps_0p5);
	y = _mm_sub_ps(y, tmp);
	y = _mm_add_ps(y, *(v4sf *)_ps_1);

	/* Evaluate the second polynom  (Pi/4 <= x <= 0) */

	v4sf y2 = *(v4sf *)_ps_sincof_p0;
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_add_ps(y2, *(v4sf *)_ps_sincof_p1);
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_add_ps(y2, *(v4sf *)_ps_sincof_p2);
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_mul_ps(y2, x);
	y2 = _mm_add_ps(y2, x);

	/* select the correct result from the two polynoms */
	xmm3 = poly_mask;
	y2 = _mm_and_ps(xmm3, y2); //, xmm3);
	y = _mm_andnot_ps(xmm3, y);
	y = _mm_add_ps(y, y2);
	/* update the sign */
	y = _mm_xor_ps(y, sign_bit);
	return y;
}

/* almost the same as sin_ps */
v4sf cos_ps(v4sf x)
{ // any x
	v4sf xmm1, xmm2 = _mm_setzero_ps(), xmm3, y;
#ifdef USE_SSE2
	v4si emm0, emm2;
#else
	v2si mm0, mm1, mm2, mm3;
#endif
	/* take the absolute value */
	x = _mm_and_ps(x, *(v4sf *)_ps_inv_sign_mask);

	/* scale by 4/Pi */
	y = _mm_mul_ps(x, *(v4sf *)_ps_cephes_FOPI);

#ifdef USE_SSE2
	/* store the integer part of y in mm0 */
	emm2 = _mm_cvttps_epi32(y);
	/* j=(j+1) & (~1) (see the cephes sources) */
	emm2 = _mm_add_epi32(emm2, *(v4si *)_pi32_1);
	emm2 = _mm_and_si128(emm2, *(v4si *)_pi32_inv1);
	y = _mm_cvtepi32_ps(emm2);

	emm2 = _mm_sub_epi32(emm2, *(v4si *)_pi32_2);

	/* get the swap sign flag */
	emm0 = _mm_andnot_si128(emm2, *(v4si *)_pi32_4);
	emm0 = _mm_slli_epi32(emm0, 29);
	/* get the polynom selection mask */
	emm2 = _mm_and_si128(emm2, *(v4si *)_pi32_2);
	emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());

	v4sf sign_bit = _mm_castsi128_ps(emm0);
	v4sf poly_mask = _mm_castsi128_ps(emm2);
#else
	/* store the integer part of y in mm0:mm1 */
	xmm2 = _mm_movehl_ps(xmm2, y);
	mm2 = _mm_cvttps_pi32(y);
	mm3 = _mm_cvttps_pi32(xmm2);

	/* j=(j+1) & (~1) (see the cephes sources) */
	mm2 = _mm_add_pi32(mm2, *(v2si *)_pi32_1);
	mm3 = _mm_add_pi32(mm3, *(v2si *)_pi32_1);
	mm2 = _mm_and_si64(mm2, *(v2si *)_pi32_inv1);
	mm3 = _mm_and_si64(mm3, *(v2si *)_pi32_inv1);

	y = _mm_cvtpi32x2_ps(mm2, mm3);

	mm2 = _mm_sub_pi32(mm2, *(v2si *)_pi32_2);
	mm3 = _mm_sub_pi32(mm3, *(v2si *)_pi32_2);

	/* get the swap sign flag in mm0:mm1 and the
	   polynom selection mask in mm2:mm3 */

	mm0 = _mm_andnot_si64(mm2, *(v2si *)_pi32_4);
	mm1 = _mm_andnot_si64(mm3, *(v2si *)_pi32_4);
	mm0 = _mm_slli_pi32(mm0, 29);
	mm1 = _mm_slli_pi32(mm1, 29);

	mm2 = _mm_and_si64(mm2, *(v2si *)_pi32_2);
	mm3 = _mm_and_si64(mm3, *(v2si *)_pi32_2);

	mm2 = _mm_cmpeq_pi32(mm2, _mm_setzero_si64());
	mm3 = _mm_cmpeq_pi32(mm3, _mm_setzero_si64());

	v4sf sign_bit, poly_mask;
	COPY_MM_TO_XMM(mm0, mm1, sign_bit);
	COPY_MM_TO_XMM(mm2, mm3, poly_mask);
	_mm_empty(); /* good-bye mmx */
#endif
	/* The magic pass: "Extended precision modular arithmetic"
	   x = ((x - y * DP1) - y * DP2) - y * DP3; */
	xmm1 = *(v4sf *)_ps_minus_cephes_DP1;
	xmm2 = *(v4sf *)_ps_minus_cephes_DP2;
	xmm3 = *(v4sf *)_ps_minus_cephes_DP3;
	xmm1 = _mm_mul_ps(y, xmm1);
	xmm2 = _mm_mul_ps(y, xmm2);
	xmm3 = _mm_mul_ps(y, xmm3);
	x = _mm_add_ps(x, xmm1);
	x = _mm_add_ps(x, xmm2);
	x = _mm_add_ps(x, xmm3);

	/* Evaluate the first polynom  (0 <= x <= Pi/4) */
	y = *(v4sf *)_ps_coscof_p0;
	v4sf z = _mm_mul_ps(x, x);

	y = _mm_mul_ps(y, z);
	y = _mm_add_ps(y, *(v4sf *)_ps_coscof_p1);
	y = _mm_mul_ps(y, z);
	y = _mm_add_ps(y, *(v4sf *)_ps_coscof_p2);
	y = _mm_mul_ps(y, z);
	y = _mm_mul_ps(y, z);
	v4sf tmp = _mm_mul_ps(z, *(v4sf *)_ps_0p5);
	y = _mm_sub_ps(y, tmp);
	y = _mm_add_ps(y, *(v4sf *)_ps_1);

	/* Evaluate the second polynom  (Pi/4 <= x <= 0) */

	v4sf y2 = *(v4sf *)_ps_sincof_p0;
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_add_ps(y2, *(v4sf *)_ps_sincof_p1);
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_add_ps(y2, *(v4sf *)_ps_sincof_p2);
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_mul_ps(y2, x);
	y2 = _mm_add_ps(y2, x);

	/* select the correct result from the two polynoms */
	xmm3 = poly_mask;
	y2 = _mm_and_ps(xmm3, y2); //, xmm3);
	y = _mm_andnot_ps(xmm3, y);
	y = _mm_add_ps(y, y2);
	/* update the sign */
	y = _mm_xor_ps(y, sign_bit);

	return y;
}

/* since sin_ps and cos_ps are almost identical, sincos_ps could replace both of them..
   it is almost as fast, and gives you a free cosine with your sine */
void sincos_ps(v4sf x, v4sf *s, v4sf *c)
{
	v4sf xmm1, xmm2, xmm3 = _mm_setzero_ps(), sign_bit_sin, y;
#ifdef USE_SSE2
	v4si emm0, emm2, emm4;
#else
	v2si mm0, mm1, mm2, mm3, mm4, mm5;
#endif
	sign_bit_sin = x;
	/* take the absolute value */
	x = _mm_and_ps(x, *(v4sf *)_ps_inv_sign_mask);
	/* extract the sign bit (upper one) */
	sign_bit_sin = _mm_and_ps(sign_bit_sin, *(v4sf *)_ps_sign_mask);

	/* scale by 4/Pi */
	y = _mm_mul_ps(x, *(v4sf *)_ps_cephes_FOPI);

#ifdef USE_SSE2
	/* store the integer part of y in emm2 */
	emm2 = _mm_cvttps_epi32(y);

	/* j=(j+1) & (~1) (see the cephes sources) */
	emm2 = _mm_add_epi32(emm2, *(v4si *)_pi32_1);
	emm2 = _mm_and_si128(emm2, *(v4si *)_pi32_inv1);
	y = _mm_cvtepi32_ps(emm2);

	emm4 = emm2;

	/* get the swap sign flag for the sine */
	emm0 = _mm_and_si128(emm2, *(v4si *)_pi32_4);
	emm0 = _mm_slli_epi32(emm0, 29);
	v4sf swap_sign_bit_sin = _mm_castsi128_ps(emm0);

	/* get the polynom selection mask for the sine*/
	emm2 = _mm_and_si128(emm2, *(v4si *)_pi32_2);
	emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());
	v4sf poly_mask = _mm_castsi128_ps(emm2);
#else
	/* store the integer part of y in mm2:mm3 */
	xmm3 = _mm_movehl_ps(xmm3, y);
	mm2 = _mm_cvttps_pi32(y);
	mm3 = _mm_cvttps_pi32(xmm3);

	/* j=(j+1) & (~1) (see the cephes sources) */
	mm2 = _mm_add_pi32(mm2, *(v2si *)_pi32_1);
	mm3 = _mm_add_pi32(mm3, *(v2si *)_pi32_1);
	mm2 = _mm_and_si64(mm2, *(v2si *)_pi32_inv1);
	mm3 = _mm_and_si64(mm3, *(v2si *)_pi32_inv1);

	y = _mm_cvtpi32x2_ps(mm2, mm3);

	mm4 = mm2;
	mm5 = mm3;

	/* get the swap sign flag for the sine */
	mm0 = _mm_and_si64(mm2, *(v2si *)_pi32_4);
	mm1 = _mm_and_si64(mm3, *(v2si *)_pi32_4);
	mm0 = _mm_slli_pi32(mm0, 29);
	mm1 = _mm_slli_pi32(mm1, 29);
	v4sf swap_sign_bit_sin;
	COPY_MM_TO_XMM(mm0, mm1, swap_sign_bit_sin);

	/* get the polynom selection mask for the sine */

	mm2 = _mm_and_si64(mm2, *(v2si *)_pi32_2);
	mm3 = _mm_and_si64(mm3, *(v2si *)_pi32_2);
	mm2 = _mm_cmpeq_pi32(mm2, _mm_setzero_si64());
	mm3 = _mm_cmpeq_pi32(mm3, _mm_setzero_si64());
	v4sf poly_mask;
	COPY_MM_TO_XMM(mm2, mm3, poly_mask);
#endif

	/* The magic pass: "Extended precision modular arithmetic"
	   x = ((x - y * DP1) - y * DP2) - y * DP3; */
	xmm1 = *(v4sf *)_ps_minus_cephes_DP1;
	xmm2 = *(v4sf *)_ps_minus_cephes_DP2;
	xmm3 = *(v4sf *)_ps_minus_cephes_DP3;
	xmm1 = _mm_mul_ps(y, xmm1);
	xmm2 = _mm_mul_ps(y, xmm2);
	xmm3 = _mm_mul_ps(y, xmm3);
	x = _mm_add_ps(x, xmm1);
	x = _mm_add_ps(x, xmm2);
	x = _mm_add_ps(x, xmm3);

#ifdef USE_SSE2
	emm4 = _mm_sub_epi32(emm4, *(v4si *)_pi32_2);
	emm4 = _mm_andnot_si128(emm4, *(v4si *)_pi32_4);
	emm4 = _mm_slli_epi32(emm4, 29);
	v4sf sign_bit_cos = _mm_castsi128_ps(emm4);
#else
	/* get the sign flag for the cosine */
	mm4 = _mm_sub_pi32(mm4, *(v2si *)_pi32_2);
	mm5 = _mm_sub_pi32(mm5, *(v2si *)_pi32_2);
	mm4 = _mm_andnot_si64(mm4, *(v2si *)_pi32_4);
	mm5 = _mm_andnot_si64(mm5, *(v2si *)_pi32_4);
	mm4 = _mm_slli_pi32(mm4, 29);
	mm5 = _mm_slli_pi32(mm5, 29);
	v4sf sign_bit_cos;
	COPY_MM_TO_XMM(mm4, mm5, sign_bit_cos);
	_mm_empty(); /* good-bye mmx */
#endif

	sign_bit_sin = _mm_xor_ps(sign_bit_sin, swap_sign_bit_sin);

	/* Evaluate the first polynom  (0 <= x <= Pi/4) */
	v4sf z = _mm_mul_ps(x, x);
	y = *(v4sf *)_ps_coscof_p0;

	y = _mm_mul_ps(y, z);
	y = _mm_add_ps(y, *(v4sf *)_ps_coscof_p1);
	y = _mm_mul_ps(y, z);
	y = _mm_add_ps(y, *(v4sf *)_ps_coscof_p2);
	y = _mm_mul_ps(y, z);
	y = _mm_mul_ps(y, z);
	v4sf tmp = _mm_mul_ps(z, *(v4sf *)_ps_0p5);
	y = _mm_sub_ps(y, tmp);
	y = _mm_add_ps(y, *(v4sf *)_ps_1);

	/* Evaluate the second polynom  (Pi/4 <= x <= 0) */

	v4sf y2 = *(v4sf *)_ps_sincof_p0;
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_add_ps(y2, *(v4sf *)_ps_sincof_p1);
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_add_ps(y2, *(v4sf *)_ps_sincof_p2);
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_mul_ps(y2, x);
	y2 = _mm_add_ps(y2, x);

	/* select the correct result from the two polynoms */
	xmm3 = poly_mask;
	v4sf ysin2 = _mm_and_ps(xmm3, y2);
	v4sf ysin1 = _mm_andnot_ps(xmm3, y);
	y2 = _mm_sub_ps(y2, ysin2);
	y = _mm_sub_ps(y, ysin1);

	xmm1 = _mm_add_ps(ysin1, ysin2);
	xmm2 = _mm_add_ps(y, y2);

	/* update the sign */
	*s = _mm_xor_ps(xmm1, sign_bit_sin);
	*c = _mm_xor_ps(xmm2, sign_bit_cos);
}
//////////////////////////

Ray::CameraParams::CameraParams(const rfw::CameraView &view, uint samples, float epsilon, uint width, uint height)
{
	pos_lensSize = glm::vec4(view.pos, view.aperture);
	right_spreadAngle = vec4(view.p2 - view.p1, view.spreadAngle);
	up = vec4(view.p3 - view.p1, 1.0f);
	p1 = vec4(view.p1, 1.0f);

	samplesTaken = samples;
	geometryEpsilon = 1e-5f;
	scrwidth = width;
	scrheight = height;
}

Ray Ray::generateFromView(const Ray::CameraParams &camera, int x, int y, float r0, float r1, float r2, float r3)
{
	Ray ray;
	const float blade = int(r0 * 9);
	r2 = (r2 - blade * (1.0f / 9.0f)) * 9.0f;
	float x1, y1, x2, y2;
	constexpr float piOver4point5 = 3.14159265359f / 4.5f;
	float bladeParam = blade * piOver4point5;
	x1 = cos(bladeParam);
	y1 = sin(bladeParam);
	bladeParam = (blade + 1.0f) * piOver4point5;
	x2 = cos(bladeParam);
	y2 = sin(bladeParam);
	if ((r2 + r3) > 1.0f)
	{
		r2 = 1.0f - r2;
		r3 = 1.0f - r3;
	}
	const float xr = x1 * r2 + x2 * r3;
	const float yr = y1 * r2 + y2 * r3;
	const vec3 p1 = camera.p1;
	const vec3 right = camera.right_spreadAngle;
	const vec3 up = camera.up;

	ray.origin = vec3(camera.pos_lensSize) + camera.pos_lensSize.w * (right * xr + up * yr);
	const float u = (float(x) + r0) * (1.0f / float(camera.scrwidth));
	const float v = (float(y) + r1) * (1.0f / float(camera.scrheight));
	const vec3 pointOnPixel = p1 + u * right + v * up;
	ray.direction = normalize(pointOnPixel - ray.origin);

	return ray;
}

RTCRayHit4 Ray::GenerateRay4(const CameraParams &camera, const int x[4], const int y[4], rfw::utils::RandomGenerator *rng)
{
	RTCRayHit4 query{};

	for (int i = 0; i < 4; i++)
	{
		query.ray.tfar[i] = 1e34f;
		query.ray.tnear[i] = 1e-5f;
		query.hit.geomID[i] = RTC_INVALID_GEOMETRY_ID;
		query.hit.primID[i] = RTC_INVALID_GEOMETRY_ID;
		query.hit.instID[0][i] = RTC_INVALID_GEOMETRY_ID;
		query.ray.id[i] = camera.scrwidth * y[i] + x[i];
	}

	static const __m128 one4 = _mm_set1_ps(1.0f);

	const __m128 r04 = _mm_set_ps(rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand());
	const __m128 r14 = _mm_set_ps(rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand());
	__m128 r24 = _mm_set_ps(rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand());
	__m128 r34 = _mm_set_ps(rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand());

	const __m128 blade4 = _mm_mul_ps(r04, _mm_set1_ps(9.0f));

	r24 = _mm_sub_ps(r24, _mm_mul_ps(_mm_mul_ps(blade4, _mm_set1_ps(1.0f / 9.0f)), _mm_set1_ps(9.0f)));

	// const float blade = float(int(r0 * 9));
	// r2 = (r2 - blade * (1.0f / 9.0f)) * 9.0f;
	// float x1, y1, x2, y2;
	constexpr float piOver4point5 = 3.14159265359f / 4.5f;
	static const __m128 piOver4point5_4 = _mm_set1_ps(piOver4point5);
	// float bladeParam = blade * piOver4point5;
	const __m128 bladeParam14 = _mm_mul_ps(blade4, piOver4point5_4);

	// x1 = cos(bladeParam);
	// y1 = sin(bladeParam);
	__m128 x14, y14;
	sincos_ps(bladeParam14, &x14, &y14);

	// bladeParam = (blade + 1.0f) * piOver4point5;
	const __m128 bladeParam24 = _mm_mul_ps(_mm_add_ps(blade4, one4), piOver4point5_4);
	// x2 = cos(bladeParam);
	// y2 = sin(bladeParam);
	__m128 x24, y24;
	sincos_ps(bladeParam24, &x24, &y24);

	// if ((r2 + r3) > 1.0f)
	const __m128i mask = _mm_castps_si128(_mm_cmpgt_ps(_mm_add_ps(r24, r34), one4));
	// r2 = 1.0f - r2;
	// r3 = 1.0f - r3;
	_mm_maskstore_ps(reinterpret_cast<float *>(&r24), mask, _mm_sub_ps(one4, r24));
	_mm_maskstore_ps(reinterpret_cast<float *>(&r34), mask, _mm_sub_ps(one4, r34));

	// const float xr = x1 * r2 + x2 * r3;
	// const float yr = y1 * r2 + y2 * r3;
	const __m128 xr4 = _mm_add_ps(_mm_mul_ps(x14, r24), _mm_mul_ps(x24, r34));
	const __m128 yr4 = _mm_add_ps(_mm_mul_ps(y14, r24), _mm_mul_ps(y24, r34));

	// ray.origin = vec3(camera.pos_lensSize) + camera.pos_lensSize.w * (right * xr + up * yr);
	const __m128 lens_size4 = _mm_set1_ps(camera.pos_lensSize.w);
	const __m128 org_x4 =
		_mm_add_ps(_mm_set1_ps(camera.pos_lensSize.x),
				   _mm_mul_ps(lens_size4, _mm_add_ps(_mm_mul_ps(_mm_set1_ps(camera.right_spreadAngle.x), xr4), _mm_mul_ps(_mm_set1_ps(camera.up.x), yr4))));
	const __m128 org_y4 =
		_mm_add_ps(_mm_set1_ps(camera.pos_lensSize.y),
				   _mm_mul_ps(lens_size4, _mm_add_ps(_mm_mul_ps(_mm_set1_ps(camera.right_spreadAngle.y), xr4), _mm_mul_ps(_mm_set1_ps(camera.up.y), yr4))));
	const __m128 org_z4 =
		_mm_add_ps(_mm_set1_ps(camera.pos_lensSize.z),
				   _mm_mul_ps(lens_size4, _mm_add_ps(_mm_mul_ps(_mm_set1_ps(camera.right_spreadAngle.z), xr4), _mm_mul_ps(_mm_set1_ps(camera.up.z), yr4))));

	// const float u = (float(x) + r0) * (1.0f / float(camera.scrwidth));
	// const float v = (float(y) + r1) * (1.0f / float(camera.scrheight));
	__m128 u4 = _mm_setr_ps(x[0], x[1], x[2], x[3]);
	__m128 v4 = _mm_setr_ps(y[0], y[1], y[2], y[3]);

	u4 = _mm_add_ps(u4, r04);
	v4 = _mm_add_ps(v4, r14);

	const __m128 scrwidth4 = _mm_set1_ps(1.0f / float(camera.scrwidth));
	const __m128 scrheight4 = _mm_set1_ps(1.0f / float(camera.scrheight));

	u4 = _mm_mul_ps(u4, scrwidth4);
	v4 = _mm_mul_ps(v4, scrheight4);

	const __m128 p1_x4 = _mm_set1_ps(camera.p1.x);
	const __m128 p1_y4 = _mm_set1_ps(camera.p1.y);
	const __m128 p1_z4 = _mm_set1_ps(camera.p1.z);

	const __m128 right_x4 = _mm_set1_ps(camera.right_spreadAngle.x);
	const __m128 right_y4 = _mm_set1_ps(camera.right_spreadAngle.y);
	const __m128 right_z4 = _mm_set1_ps(camera.right_spreadAngle.z);

	const __m128 up_x4 = _mm_set1_ps(camera.up.x);
	const __m128 up_y4 = _mm_set1_ps(camera.up.y);
	const __m128 up_z4 = _mm_set1_ps(camera.up.z);

	// const vec3 pointOnPixel = p1 + u * right + v * up;
	const __m128 pixel_x4 = _mm_add_ps(p1_x4, _mm_add_ps(_mm_mul_ps(u4, right_x4), _mm_mul_ps(v4, up_x4)));
	const __m128 pixel_y4 = _mm_add_ps(p1_y4, _mm_add_ps(_mm_mul_ps(u4, right_y4), _mm_mul_ps(v4, up_y4)));
	const __m128 pixel_z4 = _mm_add_ps(p1_z4, _mm_add_ps(_mm_mul_ps(u4, right_z4), _mm_mul_ps(v4, up_z4)));

	__m128 dir_x4 = _mm_sub_ps(pixel_x4, org_x4);
	__m128 dir_y4 = _mm_sub_ps(pixel_y4, org_y4);
	__m128 dir_z4 = _mm_sub_ps(pixel_z4, org_z4);

	__m128 length_squared_4 = _mm_mul_ps(dir_x4, dir_x4);
	length_squared_4 = _mm_add_ps(_mm_mul_ps(dir_y4, dir_y4), length_squared_4);
	length_squared_4 = _mm_add_ps(_mm_mul_ps(dir_z4, dir_z4), length_squared_4);

	const __m128 inv_length = _mm_div_ps(one4, _mm_sqrt_ps(length_squared_4));
	dir_x4 = _mm_mul_ps(dir_x4, inv_length);
	dir_y4 = _mm_mul_ps(dir_y4, inv_length);
	dir_z4 = _mm_mul_ps(dir_z4, inv_length);

	memcpy(query.ray.org_x, &org_x4, 4 * sizeof(float));
	memcpy(query.ray.org_y, &org_y4, 4 * sizeof(float));
	memcpy(query.ray.org_z, &org_z4, 4 * sizeof(float));

	memcpy(query.ray.dir_x, &dir_x4, 4 * sizeof(float));
	memcpy(query.ray.dir_y, &dir_y4, 4 * sizeof(float));
	memcpy(query.ray.dir_z, &dir_z4, 4 * sizeof(float));

	return query;
}

RTCRayHit8 Ray::GenerateRay8(const CameraParams &camera, const int x[8], const int y[8], rfw::utils::RandomGenerator *rng)
{
	RTCRayHit8 query{};

	for (int i = 0; i < 8; i++)
	{
		query.ray.tfar[i] = 1e34f;
		query.ray.tnear[i] = 1e-5f;
		query.hit.geomID[i] = RTC_INVALID_GEOMETRY_ID;
		query.hit.primID[i] = RTC_INVALID_GEOMETRY_ID;
		query.hit.instID[0][i] = RTC_INVALID_GEOMETRY_ID;
		query.ray.id[i] = camera.scrwidth * y[i] + x[i];
	}

	const __m256 one8 = _mm256_set1_ps(1.0f);

	union {
		__m256 r04;
		float r0[8];
	};
	union {
		__m256 r14;
		float r1[8];
	};
	union {
		__m256 r24;
		float r2[8];
	};
	union {
		__m256 r34;
		float r3[8];
	};

	r04 = _mm256_set_ps(rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand());
	r14 = _mm256_set_ps(rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand());
	r24 = _mm256_set_ps(rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand());
	r34 = _mm256_set_ps(rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand());

	const __m256 blade4 = _mm256_mul_ps(r04, _mm256_set1_ps(9.0f));

	r24 = _mm256_sub_ps(r24, _mm256_mul_ps(_mm256_mul_ps(blade4, _mm256_set1_ps(1.0f / 9.0f)), _mm256_set1_ps(9.0f)));

	// const float blade = float(int(r0 * 9));
	// r2 = (r2 - blade * (1.0f / 9.0f)) * 9.0f;
	// float x1, y1, x2, y2;
	constexpr float piOver4point5 = 3.14159265359f / 4.5f;
	const __m256 piOver4point5_4 = _mm256_set1_ps(piOver4point5);
	// float bladeParam = blade * piOver4point5;
	const __m256 bladeParam14 = _mm256_mul_ps(blade4, piOver4point5_4);

	// x1 = cos(bladeParam);
	// y1 = sin(bladeParam);
	union {
		__m256 x14;
		__m128 x1_4[2];
	};
	union {
		__m256 y14;
		__m128 y1_4[2];
	};
	sincos_ps(_mm256_extractf128_ps(bladeParam14, 0), &x1_4[0], &y1_4[0]);
	sincos_ps(_mm256_extractf128_ps(bladeParam14, 1), &x1_4[1], &y1_4[1]);

	// bladeParam = (blade + 1.0f) * piOver4point5;
	const __m256 bladeParam24 = _mm256_mul_ps(_mm256_add_ps(blade4, one8), piOver4point5_4);
	// x2 = cos(bladeParam);
	// y2 = sin(bladeParam);
	union {
		__m256 x24;
		__m128 x2_4[2];
	};
	union {
		__m256 y24;
		__m128 y2_4[2];
	};
	sincos_ps(_mm256_extractf128_ps(bladeParam24, 0), &x2_4[0], &y2_4[0]);
	sincos_ps(_mm256_extractf128_ps(bladeParam24, 1), &x2_4[1], &y2_4[1]);

	// if ((r2 + r3) > 1.0f)
	const __m128 one4 = _mm_set1_ps(1.0f);
	const __m128i mask1 = _mm_castps_si128(_mm_cmpgt_ps(_mm_add_ps(_mm256_extractf128_ps(r24, 0), _mm256_extractf128_ps(r34, 0)), one4));
	const __m128i mask2 = _mm_castps_si128(_mm_cmpgt_ps(_mm_add_ps(_mm256_extractf128_ps(r24, 1), _mm256_extractf128_ps(r34, 1)), one4));
	// r2 = 1.0f - r2;
	// r3 = 1.0f - r3;
	_mm_maskstore_ps(r2, mask1, _mm_sub_ps(one4, _mm256_extractf128_ps(r24, 0)));
	_mm_maskstore_ps(r3, mask2, _mm_sub_ps(one4, _mm256_extractf128_ps(r34, 0)));
	_mm_maskstore_ps(r2 + 4, mask1, _mm_sub_ps(one4, _mm256_extractf128_ps(r24, 1)));
	_mm_maskstore_ps(r3 + 4, mask2, _mm_sub_ps(one4, _mm256_extractf128_ps(r34, 1)));

	// const float xr = x1 * r2 + x2 * r3;
	// const float yr = y1 * r2 + y2 * r3;
	const __m256 xr4 = _mm256_add_ps(_mm256_mul_ps(x14, r24), _mm256_mul_ps(x24, r34));
	const __m256 yr4 = _mm256_add_ps(_mm256_mul_ps(y14, r24), _mm256_mul_ps(y24, r34));

	union {
		__m256 org_x4;
		float org_x[8];
	};
	union {
		__m256 org_y4;
		float org_y[8];
	};
	union {
		__m256 org_z4;
		float org_z[8];
	};

	// ray.origin = vec3(camera.pos_lensSize) + camera.pos_lensSize.w * (right * xr + up * yr);
	const __m256 lens_size4 = _mm256_set1_ps(camera.pos_lensSize.w);
	org_x4 = _mm256_add_ps(_mm256_set1_ps(camera.pos_lensSize.x),
						   _mm256_mul_ps(lens_size4, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(camera.right_spreadAngle.x), xr4),
																   _mm256_mul_ps(_mm256_set1_ps(camera.up.x), yr4))));
	org_y4 = _mm256_add_ps(_mm256_set1_ps(camera.pos_lensSize.y),
						   _mm256_mul_ps(lens_size4, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(camera.right_spreadAngle.y), yr4),
																   _mm256_mul_ps(_mm256_set1_ps(camera.up.y), yr4))));
	org_z4 = _mm256_add_ps(_mm256_set1_ps(camera.pos_lensSize.z),
						   _mm256_mul_ps(lens_size4, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(camera.right_spreadAngle.z), yr4),
																   _mm256_mul_ps(_mm256_set1_ps(camera.up.z), yr4))));

	// const float u = (float(x) + r0) * (1.0f / float(camera.scrwidth));
	// const float v = (float(y) + r1) * (1.0f / float(camera.scrheight));
	__m256 u4 = _mm256_setr_ps(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
	__m256 v4 = _mm256_setr_ps(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]);

	u4 = _mm256_add_ps(u4, r04);
	v4 = _mm256_add_ps(v4, r14);

	const __m256 scrwidth4 = _mm256_set1_ps(1.0f / float(camera.scrwidth));
	const __m256 scrheight4 = _mm256_set1_ps(1.0f / float(camera.scrheight));

	u4 = _mm256_mul_ps(u4, scrwidth4);
	v4 = _mm256_mul_ps(v4, scrheight4);

	union {
		__m256 pixel_x4;
		float pixel_x[8];
	};
	union {
		__m256 pixel_y4;
		float pixel_y[8];
	};
	union {
		__m256 pixel_z4;
		float pixel_z[8];
	};

	const __m256 p1_x4 = _mm256_set1_ps(camera.p1.x);
	const __m256 p1_y4 = _mm256_set1_ps(camera.p1.y);
	const __m256 p1_z4 = _mm256_set1_ps(camera.p1.z);
	const __m256 right_x4 = _mm256_set1_ps(camera.right_spreadAngle.x);
	const __m256 right_y4 = _mm256_set1_ps(camera.right_spreadAngle.y);
	const __m256 right_z4 = _mm256_set1_ps(camera.right_spreadAngle.z);
	const __m256 up_x4 = _mm256_set1_ps(camera.up.x);
	const __m256 up_y4 = _mm256_set1_ps(camera.up.y);
	const __m256 up_z4 = _mm256_set1_ps(camera.up.z);

	// const vec3 pointOnPixel = p1 + u * right + v * up;
	pixel_x4 = _mm256_add_ps(p1_x4, _mm256_add_ps(_mm256_mul_ps(u4, right_x4), _mm256_mul_ps(v4, up_x4)));
	pixel_y4 = _mm256_add_ps(p1_y4, _mm256_add_ps(_mm256_mul_ps(u4, right_y4), _mm256_mul_ps(v4, up_y4)));
	pixel_z4 = _mm256_add_ps(p1_z4, _mm256_add_ps(_mm256_mul_ps(u4, right_z4), _mm256_mul_ps(v4, up_z4)));

	__m256 dir_x4;
	__m256 dir_y4;
	__m256 dir_z4;

	dir_x4 = _mm256_sub_ps(pixel_x4, org_x4);
	dir_y4 = _mm256_sub_ps(pixel_y4, org_y4);
	dir_z4 = _mm256_sub_ps(pixel_z4, org_z4);

	__m256 length_squared_4 = _mm256_mul_ps(dir_x4, dir_x4);
	length_squared_4 = _mm256_add_ps(_mm256_mul_ps(dir_y4, dir_y4), length_squared_4);
	length_squared_4 = _mm256_add_ps(_mm256_mul_ps(dir_z4, dir_z4), length_squared_4);

	const __m256 inv_length = _mm256_div_ps(one8, _mm256_sqrt_ps(length_squared_4));
	dir_x4 = _mm256_mul_ps(dir_x4, inv_length);
	dir_y4 = _mm256_mul_ps(dir_y4, inv_length);
	dir_z4 = _mm256_mul_ps(dir_z4, inv_length);

	_mm256_store_ps(query.ray.org_x, org_x4);
	_mm256_store_ps(query.ray.org_y, org_y4);
	_mm256_store_ps(query.ray.org_z, org_z4);

	_mm256_store_ps(query.ray.dir_x, dir_x4);
	_mm256_store_ps(query.ray.dir_y, dir_y4);
	_mm256_store_ps(query.ray.dir_z, dir_z4);

	return query;
}

RTCRayHit16 Ray::GenerateRay16(const CameraParams &camera, const int x[16], const int y[16], rfw::utils::RandomGenerator *rng)
{
	RTCRayHit16 query{};

	for (int i = 0; i < 16; i++)
	{
		query.ray.tfar[i] = 1e34f;
		query.ray.tnear[i] = 1e-5f;
		query.hit.geomID[i] = RTC_INVALID_GEOMETRY_ID;
		query.hit.primID[i] = RTC_INVALID_GEOMETRY_ID;
		query.hit.instID[0][i] = RTC_INVALID_GEOMETRY_ID;
		query.ray.id[i] = camera.scrwidth * y[i] + x[i];
	}

	const __m256 one8 = _mm256_set1_ps(1.0f);

	union {
		__m256 r0_16[2];
		float r0[16];
	};
	union {
		__m256 r1_16[2];
		float r1[16];
	};
	union {
		__m256 r2_16[2];
		float r2[16];
	};
	union {
		__m256 r3_16[2];
		float r3[16];
	};

	for (int i = 0; i < 16; i++)
	{
		r0[i] = rng->Rand();
		r1[i] = rng->Rand();
		r2[i] = rng->Rand();
		r3[i] = rng->Rand();
	}

	const __m256 blade0_8 = _mm256_mul_ps(r0_16[0], _mm256_set1_ps(9.0f));
	const __m256 blade1_8 = _mm256_mul_ps(r0_16[1], _mm256_set1_ps(9.0f));
	static const __m256 one_over_9 = _mm256_set1_ps(1.0f / 9.0f);
	static const __m256 nine = _mm256_set1_ps(9.0f);

	// r2 = (r2 - blade * (1.0f / 9.0f)) * 9.0f;
	r2_16[0] = _mm256_sub_ps(r2_16[0], _mm256_mul_ps(_mm256_mul_ps(blade0_8, one_over_9), nine));
	r2_16[1] = _mm256_sub_ps(r2_16[1], _mm256_mul_ps(_mm256_mul_ps(blade1_8, one_over_9), nine));

	// const float blade = float(int(r0 * 9));
	// r2 = (r2 - blade * (1.0f / 9.0f)) * 9.0f;
	// float x1, y1, x2, y2;
	static const __m256 piOver4point5_4 = _mm256_set1_ps(3.14159265359f / 4.5f);
	// float bladeParam = blade * piOver4point5;
	const __m256 bladeParam1_0 = _mm256_mul_ps(blade0_8, piOver4point5_4);
	const __m256 bladeParam1_1 = _mm256_mul_ps(blade1_8, piOver4point5_4);

	union {
		__m256 x1_8[2];
		__m128 x1_4[4];
	};
	union {
		__m256 y1_8[2];
		__m128 y1_4[4];
	};
	union {
		__m256 x2_8[2];
		__m128 x2_4[4];
	};
	union {
		__m256 y2_8[2];
		__m128 y2_4[4];
	};

	// x1 = cos(bladeParam);
	// y1 = sin(bladeParam);
	sincos_ps(_mm256_extractf128_ps(bladeParam1_0, 0), &x1_4[0], &y1_4[0]);
	sincos_ps(_mm256_extractf128_ps(bladeParam1_0, 1), &x1_4[1], &y1_4[1]);
	sincos_ps(_mm256_extractf128_ps(bladeParam1_1, 0), &x1_4[2], &y1_4[2]);
	sincos_ps(_mm256_extractf128_ps(bladeParam1_1, 1), &x1_4[3], &y1_4[3]);

	// bladeParam = (blade + 1.0f) * piOver4point5;
	const __m256 bladeParam2_0 = _mm256_mul_ps(_mm256_add_ps(blade0_8, one8), piOver4point5_4);
	const __m256 bladeParam2_1 = _mm256_mul_ps(_mm256_add_ps(blade1_8, one8), piOver4point5_4);
	// x2 = cos(bladeParam);
	// y2 = sin(bladeParam);
	sincos_ps(_mm256_extractf128_ps(bladeParam2_0, 0), &x2_4[0], &y2_4[0]);
	sincos_ps(_mm256_extractf128_ps(bladeParam2_0, 1), &x2_4[1], &y2_4[1]);
	sincos_ps(_mm256_extractf128_ps(bladeParam2_1, 0), &x2_4[2], &y2_4[2]);
	sincos_ps(_mm256_extractf128_ps(bladeParam2_1, 1), &x2_4[3], &y2_4[3]);

	// if ((r2 + r3) > 1.0f)
	const __m128 one4 = _mm_set1_ps(1.0f);
	const __m128i mask0_1 = _mm_castps_si128(_mm_cmpgt_ps(_mm_add_ps(_mm256_extractf128_ps(r2_16[0], 0), _mm256_extractf128_ps(r3_16[0], 0)), one4));
	const __m128i mask0_2 = _mm_castps_si128(_mm_cmpgt_ps(_mm_add_ps(_mm256_extractf128_ps(r2_16[0], 1), _mm256_extractf128_ps(r3_16[0], 1)), one4));
	const __m128i mask1_1 = _mm_castps_si128(_mm_cmpgt_ps(_mm_add_ps(_mm256_extractf128_ps(r2_16[1], 0), _mm256_extractf128_ps(r3_16[1], 0)), one4));
	const __m128i mask1_2 = _mm_castps_si128(_mm_cmpgt_ps(_mm_add_ps(_mm256_extractf128_ps(r2_16[1], 1), _mm256_extractf128_ps(r3_16[1], 1)), one4));

	// r2 = 1.0f - r2;
	// r3 = 1.0f - r3;
	_mm_maskstore_ps(r2, mask0_1, _mm_sub_ps(one4, _mm256_extractf128_ps(r2_16[0], 0)));
	_mm_maskstore_ps(r3, mask0_2, _mm_sub_ps(one4, _mm256_extractf128_ps(r3_16[0], 0)));
	_mm_maskstore_ps(r2 + 4, mask0_1, _mm_sub_ps(one4, _mm256_extractf128_ps(r2_16[0], 1)));
	_mm_maskstore_ps(r3 + 4, mask0_2, _mm_sub_ps(one4, _mm256_extractf128_ps(r3_16[0], 1)));

	_mm_maskstore_ps(r2 + 8, mask1_1, _mm_sub_ps(one4, _mm256_extractf128_ps(r2_16[1], 0)));
	_mm_maskstore_ps(r3 + 8, mask1_2, _mm_sub_ps(one4, _mm256_extractf128_ps(r3_16[1], 0)));
	_mm_maskstore_ps(r2 + 12, mask1_1, _mm_sub_ps(one4, _mm256_extractf128_ps(r2_16[1], 1)));
	_mm_maskstore_ps(r3 + 12, mask1_2, _mm_sub_ps(one4, _mm256_extractf128_ps(r3_16[1], 1)));

	__m256 xr_8[2];
	__m256 yr_8[2];

	// const float xr = x1 * r2 + x2 * r3;
	// const float yr = y1 * r2 + y2 * r3;
	xr_8[0] = _mm256_add_ps(_mm256_mul_ps(x1_8[0], r2_16[0]), _mm256_mul_ps(x2_8[0], r3_16[0]));
	yr_8[0] = _mm256_add_ps(_mm256_mul_ps(y1_8[0], r2_16[0]), _mm256_mul_ps(y2_8[0], r3_16[0]));
	xr_8[1] = _mm256_add_ps(_mm256_mul_ps(x1_8[1], r2_16[1]), _mm256_mul_ps(x2_8[1], r3_16[1]));
	yr_8[1] = _mm256_add_ps(_mm256_mul_ps(y1_8[1], r2_16[1]), _mm256_mul_ps(y2_8[1], r3_16[1]));

	union {
		__m256 org_x4[2];
		float org_x[16];
	};
	union {
		__m256 org_y4[2];
		float org_y[16];
	};
	union {
		__m256 org_z4[2];
		float org_z[16];
	};

	// ray.origin = vec3(camera.pos_lensSize) + camera.pos_lensSize.w * (right * xr + up * yr);
	const __m256 lens_size4 = _mm256_set1_ps(camera.pos_lensSize.w);

	org_x4[0] = _mm256_add_ps(_mm256_set1_ps(camera.pos_lensSize.x),
							  _mm256_mul_ps(lens_size4, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(camera.right_spreadAngle.x), xr_8[0]),
																	  _mm256_mul_ps(_mm256_set1_ps(camera.up.x), yr_8[0]))));
	org_x4[1] = _mm256_add_ps(_mm256_set1_ps(camera.pos_lensSize.x),
							  _mm256_mul_ps(lens_size4, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(camera.right_spreadAngle.x), xr_8[1]),
																	  _mm256_mul_ps(_mm256_set1_ps(camera.up.x), yr_8[1]))));
	org_y4[0] = _mm256_add_ps(_mm256_set1_ps(camera.pos_lensSize.y),
							  _mm256_mul_ps(lens_size4, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(camera.right_spreadAngle.y), xr_8[0]),
																	  _mm256_mul_ps(_mm256_set1_ps(camera.up.y), yr_8[0]))));
	org_y4[1] = _mm256_add_ps(_mm256_set1_ps(camera.pos_lensSize.y),
							  _mm256_mul_ps(lens_size4, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(camera.right_spreadAngle.y), xr_8[1]),
																	  _mm256_mul_ps(_mm256_set1_ps(camera.up.y), yr_8[1]))));
	org_z4[0] = _mm256_add_ps(_mm256_set1_ps(camera.pos_lensSize.z),
							  _mm256_mul_ps(lens_size4, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(camera.right_spreadAngle.z), xr_8[0]),
																	  _mm256_mul_ps(_mm256_set1_ps(camera.up.z), yr_8[0]))));
	org_z4[1] = _mm256_add_ps(_mm256_set1_ps(camera.pos_lensSize.z),
							  _mm256_mul_ps(lens_size4, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(camera.right_spreadAngle.z), xr_8[1]),
																	  _mm256_mul_ps(_mm256_set1_ps(camera.up.z), yr_8[1]))));

	// const float u = (float(x) + r0) * (1.0f / float(camera.scrwidth));
	// const float v = (float(y) + r1) * (1.0f / float(camera.scrheight));
	__m256 u8[2];
	u8[0] = _mm256_setr_ps(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
	u8[1] = _mm256_setr_ps(x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15]);
	__m256 v8[2];
	v8[0] = _mm256_setr_ps(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]);
	v8[1] = _mm256_setr_ps(y[8], y[9], y[10], y[11], y[12], y[13], y[14], y[15]);

	u8[0] = _mm256_add_ps(u8[0], r0_16[0]);
	u8[1] = _mm256_add_ps(u8[1], r0_16[1]);
	v8[0] = _mm256_add_ps(v8[0], r1_16[0]);
	v8[1] = _mm256_add_ps(v8[1], r1_16[1]);

	const __m256 scrwidth4 = _mm256_set1_ps(1.0f / float(camera.scrwidth));
	const __m256 scrheight4 = _mm256_set1_ps(1.0f / float(camera.scrheight));

	u8[0] = _mm256_mul_ps(u8[0], scrwidth4);
	u8[1] = _mm256_mul_ps(u8[1], scrwidth4);
	v8[0] = _mm256_mul_ps(v8[0], scrheight4);
	v8[1] = _mm256_mul_ps(v8[1], scrheight4);

	__m256 pixel_x4[2];
	__m256 pixel_y4[2];
	__m256 pixel_z4[2];

	const __m256 p1_x4 = _mm256_set1_ps(camera.p1.x);
	const __m256 p1_y4 = _mm256_set1_ps(camera.p1.y);
	const __m256 p1_z4 = _mm256_set1_ps(camera.p1.z);
	const __m256 right_x4 = _mm256_set1_ps(camera.right_spreadAngle.x);
	const __m256 right_y4 = _mm256_set1_ps(camera.right_spreadAngle.y);
	const __m256 right_z4 = _mm256_set1_ps(camera.right_spreadAngle.z);
	const __m256 up_x4 = _mm256_set1_ps(camera.up.x);
	const __m256 up_y4 = _mm256_set1_ps(camera.up.y);
	const __m256 up_z4 = _mm256_set1_ps(camera.up.z);

	// const vec3 pointOnPixel = p1 + u * right + v * up;
	pixel_x4[0] = _mm256_add_ps(p1_x4, _mm256_add_ps(_mm256_mul_ps(u8[0], right_x4), _mm256_mul_ps(v8[0], up_x4)));
	pixel_x4[1] = _mm256_add_ps(p1_x4, _mm256_add_ps(_mm256_mul_ps(u8[1], right_x4), _mm256_mul_ps(v8[1], up_x4)));
	pixel_y4[0] = _mm256_add_ps(p1_y4, _mm256_add_ps(_mm256_mul_ps(u8[0], right_y4), _mm256_mul_ps(v8[0], up_y4)));
	pixel_y4[1] = _mm256_add_ps(p1_y4, _mm256_add_ps(_mm256_mul_ps(u8[1], right_y4), _mm256_mul_ps(v8[1], up_y4)));
	pixel_z4[0] = _mm256_add_ps(p1_z4, _mm256_add_ps(_mm256_mul_ps(u8[0], right_z4), _mm256_mul_ps(v8[0], up_z4)));
	pixel_z4[1] = _mm256_add_ps(p1_z4, _mm256_add_ps(_mm256_mul_ps(u8[1], right_z4), _mm256_mul_ps(v8[1], up_z4)));

	union {
		__m256 dir_x4[2];
		float dir_x[16];
	};
	union {
		__m256 dir_y4[2];
		float dir_y[16];
	};
	union {
		__m256 dir_z4[2];
		float dir_z[16];
	};

	dir_x4[0] = _mm256_sub_ps(pixel_x4[0], org_x4[0]);
	dir_x4[1] = _mm256_sub_ps(pixel_x4[1], org_x4[1]);
	dir_y4[0] = _mm256_sub_ps(pixel_y4[0], org_y4[0]);
	dir_y4[1] = _mm256_sub_ps(pixel_y4[1], org_y4[1]);
	dir_z4[0] = _mm256_sub_ps(pixel_z4[0], org_z4[0]);
	dir_z4[1] = _mm256_sub_ps(pixel_z4[1], org_z4[1]);

	__m256 length_squared_4[2];
	length_squared_4[0] = _mm256_mul_ps(dir_x4[0], dir_x4[0]);
	length_squared_4[0] = _mm256_add_ps(_mm256_mul_ps(dir_y4[0], dir_y4[0]), length_squared_4[0]);
	length_squared_4[0] = _mm256_add_ps(_mm256_mul_ps(dir_z4[0], dir_z4[0]), length_squared_4[0]);

	length_squared_4[1] = _mm256_mul_ps(dir_x4[1], dir_x4[1]);
	length_squared_4[1] = _mm256_add_ps(_mm256_mul_ps(dir_y4[1], dir_y4[1]), length_squared_4[1]);
	length_squared_4[1] = _mm256_add_ps(_mm256_mul_ps(dir_z4[1], dir_z4[1]), length_squared_4[1]);

	__m256 inv_length[2];
	inv_length[0] = _mm256_div_ps(one8, _mm256_sqrt_ps(length_squared_4[0]));
	inv_length[1] = _mm256_div_ps(one8, _mm256_sqrt_ps(length_squared_4[1]));

	dir_x4[0] = _mm256_mul_ps(dir_x4[0], inv_length[0]);
	dir_x4[1] = _mm256_mul_ps(dir_x4[1], inv_length[1]);
	dir_y4[0] = _mm256_mul_ps(dir_y4[0], inv_length[0]);
	dir_y4[1] = _mm256_mul_ps(dir_y4[1], inv_length[1]);
	dir_z4[0] = _mm256_mul_ps(dir_z4[0], inv_length[0]);
	dir_z4[1] = _mm256_mul_ps(dir_z4[1], inv_length[1]);

	memcpy(query.ray.org_x, org_x, 16 * sizeof(float));
	memcpy(query.ray.org_y, org_y, 16 * sizeof(float));
	memcpy(query.ray.org_z, org_z, 16 * sizeof(float));

	memcpy(query.ray.dir_x, dir_x, 16 * sizeof(float));
	memcpy(query.ray.dir_y, dir_y, 16 * sizeof(float));
	memcpy(query.ray.dir_z, dir_z, 16 * sizeof(float));

	return query;
}