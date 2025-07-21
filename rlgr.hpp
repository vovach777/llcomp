#pragma once
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include "bitstream.hpp"

namespace RLGR
{

	constexpr uint32_t LSGR = 3; /* shift count to convert kp to k */
	constexpr uint32_t KPMAX = 80; /* max value for kp or krp */
	/* Constants used in RLGR1/RLGR3 algorithm */
	constexpr uint32_t UP_GR = 4;  /* increase in kp after a zero run in RL mode */
	constexpr uint32_t DN_GR = 6;  /* decrease in kp after a nonzero symbol in RL mode */
	constexpr uint32_t UQ_GR = 3;  /* increase in kp after nonzero symbol in GR mode */
	constexpr uint32_t DQ_GR = 3;  /* decrease in kp after zero symbol in GR mode */

	/*
		* Update the passed parameter and clamp it to the range [0, KPMAX]
		* Return the value of parameter right-shifted by LSGR
		*/
	inline uint32_t UpdateParam(uint32_t &param, int32_t deltaP)
	{
		if (deltaP < 0)
		{
			const uint32_t udeltaP = static_cast<uint32_t>(-deltaP);
			if (udeltaP > param)
				param = 0;
			else
				param -= udeltaP;
		}
		else
			param += static_cast<uint32_t>(deltaP);

		if ((param) > KPMAX)
			(param) = KPMAX;
		return (param) >> LSGR;
	}

	struct Encoder
	{
		uint8_t k = 1;
		uint8_t kp = 1 << LSGR;
		uint8_t krp = 1 << LSGR;

		uint32_t numZeros = 0;

		template <typename ExternalGREncoder_f>
		static void code_gr(uint32_t &krp, uint32_t val, ExternalGREncoder_f &&ExternalGREncoder_f)
		{
			uint32_t kr = krp >> LSGR;

			ExternalGREncoder_f(val, kr);
			const uint32_t vk = val >> kr;

			/* update krp, only if it is not equal to 1 */
			if (vk == 0)
			{
				(void)UpdateParam(krp, -2);
			}
			else if (vk > 1)
			{
				(void)UpdateParam(krp, static_cast<int32_t>(vk));
			}
		}

		/* Converts the input value to (2 * abs(input) - sign(input)), where sign(input) = (input < 0 ? 1 :
		 * 0) and returns it */
		inline uint32_t Get2MagSign(int32_t input)
		{
			return static_cast<uint32_t>((input >= 0) ? 2 * input : -2 * input - 1);
		}

		inline bool is_rl_pending()
		{
			return numZeros > 0;
		}

		template <typename OutputBit_f>
		void encode(int input, OutputBit_f &&OutputBit)
		{

			int input = 0;

			if (k)
			{
				/* RUN-LENGTH MODE */

				if (input == 0)
				{
					numZeros++;
					// emit output zeros
					auto runmax = 1 << k;
					if (numZeros == runmax)
					{
						OutputBit(1, 0); /* output a zero bit */
						numZeros -= runmax;
						k = UpdateParam(&kp, UP_GR); /* update kp, k */
					}
					return;
				}
				/* output a 1 to terminate runs */
				OutputBit(1, 1);

				/* output the remaining run length using k bits */
				OutputBits(k, numZeros);

				/* note: when we reach here and the last byte being encoded is 0, we still
				need to output the last two bits, otherwise mstsc will crash */

				/* encode the nonzero value using GR coding */
				const uint32_t mag = input < 0 ? -input : input;
				sign = (input < 0 ? 1 : 0); /* sign of input coefficient */

				OutputBit(1, input < 0 ? 1 : 0);									   /* output the sign bit */
				code_gr(krp, mag ? mag - 1 : 0, std::forward<OutputBit_f>(OutputBit)); /* output GR code for (mag - 1) */

				k = UpdateParam(&kp, -DN_GR);

				numZeros = 0;
			}
			else
			{
				/* GOLOMB-RICE MODE */

				/* RLGR1 variant */

				/* convert input to (2*magnitude - sign), encode using GR code */

				auto twoMs = Get2MagSign(input);
				code_gr(krp, twoMs, std::forward<OutputBit_f>(OutputBit));

				/* update k, kp */
				/* NOTE: as of Aug 2011, the algorithm is still wrongly documented
				and the update direction is reversed */
				if (twoMs)
				{
					k = UpdateParam(&kp, -DQ_GR);
				}
				else
				{
					k = UpdateParam(&kp, UQ_GR);
				}
			}
		}
	};
	struct Decoder
	{
		uint8_t k = 1;
		uint8_t kr = 1;
		uint8_t kp = 1 << LSGR;
		uint8_t krp = 1 << LSGR;

		template <typename Peek32_f, typename Skip_f>
		int rfx_rlgr_decode(
							// const uint8_t *pSrcData, size_t SrcSize,
							// uint16_t* pDstData, size_t rDstSize,
							Peek32_f && Peek32,
							Skip_f  && Skip
						)
		{
			uint32_t vk = 0;
			size_t run = 0;
			size_t cnt = 0;
			size_t size = 0;
			size_t offset = 0;
			INT16 mag = 0;
			UINT32 k = 0;
			UINT32 kp = 0;
			UINT32 kr = 0;
			UINT32 krp = 0;
			UINT16 code = 0;
			UINT32 sign = 0;
			UINT32 nIdx = 0;
			UINT32 val1 = 0;
			UINT32 val2 = 0;
			INT16 *pOutput = NULL;
			wBitStream *bs = NULL;
			wBitStream s_bs = {0};
			const SSIZE_T DstSize = rDstSize;



			pOutput = pDstData;





				if (k)
				{
					/* Run-Length (RL) Mode */

					run = 0;
					uint32_t cnt = 0;

					/* count number of leading 0s */
					while (Peek32() == 0) {
						cnt += 32;
						Skip(32);
						/* TODO: limit to prevent infinity loop */
					}
					auto ziros_in_buffer = __builtin_clz(Peek32());
					Skip(ziros_in_buffer + 1);
					uint32_t vk = cnt + ziros_in_buffer;


					while ((cnt == 32) && (BitStream_GetRemainingLength(bs) > 0))
					{
						BitStream_Shift32(bs);

						cnt = lzcnt_s(bs->accumulator);

						nbits = BitStream_GetRemainingLength(bs);

						if (cnt > nbits)
							cnt = nbits;

						WINPR_ASSERT(cnt + vk <= UINT32_MAX);
						vk += WINPR_ASSERTING_INT_CAST(uint32_t, cnt);
					}

					BitStream_Shift(bs, (vk % 32));

					if (BitStream_GetRemainingLength(bs) < 1)
						break;

					BitStream_Shift(bs, 1);

					while (vk--)
					{
						const UINT32 add = (1 << k); /* add (1 << k) to run length */
						run += add;

						/* update k, kp params */

						kp += UP_GR;

						if (kp > KPMAX)
							kp = KPMAX;

						k = kp >> LSGR;
					}

					/* next k bits contain run length remainder */

					if (BitStream_GetRemainingLength(bs) < k)
						break;

					bs->mask = ((1 << k) - 1);
					run += ((bs->accumulator >> (32 - k)) & bs->mask);
					BitStream_Shift(bs, k);

					/* read sign bit */

					if (BitStream_GetRemainingLength(bs) < 1)
						break;

					sign = (bs->accumulator & 0x80000000) ? 1 : 0;
					BitStream_Shift(bs, 1);

					/* count number of leading 1s */

					cnt = lzcnt_s(~(bs->accumulator));

					nbits = BitStream_GetRemainingLength(bs);

					if (cnt > nbits)
						cnt = nbits;

					vk = WINPR_ASSERTING_INT_CAST(uint32_t, cnt);

					while ((cnt == 32) && (BitStream_GetRemainingLength(bs) > 0))
					{
						BitStream_Shift32(bs);

						cnt = lzcnt_s(~(bs->accumulator));

						nbits = BitStream_GetRemainingLength(bs);

						if (cnt > nbits)
							cnt = nbits;

						WINPR_ASSERT(cnt + vk <= UINT32_MAX);
						vk += WINPR_ASSERTING_INT_CAST(uint32_t, cnt);
					}

					BitStream_Shift(bs, (vk % 32));

					if (BitStream_GetRemainingLength(bs) < 1)
						break;

					BitStream_Shift(bs, 1);

					/* next kr bits contain code remainder */

					if (BitStream_GetRemainingLength(bs) < kr)
						break;

					bs->mask = ((1 << kr) - 1);
					if (kr > 0)
						code = (UINT16)((bs->accumulator >> (32 - kr)) & bs->mask);
					else
						code = 0;
					BitStream_Shift(bs, kr);

					/* add (vk << kr) to code */

					code |= (vk << kr);

					if (!vk)
					{
						/* update kr, krp params */

						if (krp > 2)
							krp -= 2;
						else
							krp = 0;

						kr = krp >> LSGR;
					}
					else if (vk != 1)
					{
						/* update kr, krp params */

						krp += vk;

						if (krp > KPMAX)
							krp = KPMAX;

						kr = krp >> LSGR;
					}

					/* update k, kp params */

					if (kp > DN_GR)
						kp -= DN_GR;
					else
						kp = 0;

					k = kp >> LSGR;

					/* compute magnitude from code */

					if (sign)
						mag = WINPR_ASSERTING_INT_CAST(int16_t, (code + 1)) * -1;
					else
						mag = WINPR_ASSERTING_INT_CAST(int16_t, code + 1);

					/* write to output stream */

					offset = WINPR_ASSERTING_INT_CAST(size_t, (pOutput)-pDstData);
					size = run;

					if ((offset + size) > rDstSize)
						size = WINPR_ASSERTING_INT_CAST(size_t, DstSize) - offset;

					if (size)
					{
						ZeroMemory(pOutput, size * sizeof(INT16));
						pOutput += size;
					}

					if ((pOutput - pDstData) < DstSize)
					{
						*pOutput = mag;
						pOutput++;
					}
				}
				else
				{
					/* Golomb-Rice (GR) Mode */

					/* count number of leading 1s */

					cnt = lzcnt_s(~(bs->accumulator));

					size_t nbits = BitStream_GetRemainingLength(bs);

					if (cnt > nbits)
						cnt = nbits;

					vk = WINPR_ASSERTING_INT_CAST(uint32_t, cnt);

					while ((cnt == 32) && (BitStream_GetRemainingLength(bs) > 0))
					{
						BitStream_Shift32(bs);

						cnt = lzcnt_s(~(bs->accumulator));

						nbits = BitStream_GetRemainingLength(bs);

						if (cnt > nbits)
							cnt = nbits;

						WINPR_ASSERT(cnt + vk <= UINT32_MAX);
						vk += WINPR_ASSERTING_INT_CAST(uint32_t, cnt);
					}

					BitStream_Shift(bs, (vk % 32));

					if (BitStream_GetRemainingLength(bs) < 1)
						break;

					BitStream_Shift(bs, 1);

					/* next kr bits contain code remainder */

					if (BitStream_GetRemainingLength(bs) < kr)
						break;

					bs->mask = ((1 << kr) - 1);
					if (kr > 0)
						code = (UINT16)((bs->accumulator >> (32 - kr)) & bs->mask);
					else
						code = 0;
					BitStream_Shift(bs, kr);

					/* add (vk << kr) to code */

					code |= (vk << kr);

					if (!vk)
					{
						/* update kr, krp params */

						if (krp > 2)
							krp -= 2;
						else
							krp = 0;

						kr = (krp >> LSGR) & UINT32_MAX;
					}
					else if (vk != 1)
					{
						/* update kr, krp params */

						krp += vk;

						if (krp > KPMAX)
							krp = KPMAX;

						kr = krp >> LSGR;
					}

					if (mode == RLGR1) /* RLGR1 */
					{
						if (!code)
						{
							/* update k, kp params */

							kp += UQ_GR;

							if (kp > KPMAX)
								kp = KPMAX;

							k = kp >> LSGR;

							mag = 0;
						}
						else
						{
							/* update k, kp params */

							if (kp > DQ_GR)
								kp -= DQ_GR;
							else
								kp = 0;

							k = kp >> LSGR;

							/*
							 * code = 2 * mag - sign
							 * sign + code = 2 * mag
							 */

							if (code & 1)
								mag = WINPR_ASSERTING_INT_CAST(INT16, (code + 1) >> 1) * -1;
							else
								mag = WINPR_ASSERTING_INT_CAST(INT16, code >> 1);
						}

						if ((pOutput - pDstData) < DstSize)
						{
							*pOutput = mag;
							pOutput++;
						}
					}
				}
			}

			offset = WINPR_ASSERTING_INT_CAST(size_t, (pOutput - pDstData));

			if (offset < rDstSize)
			{
				size = WINPR_ASSERTING_INT_CAST(size_t, DstSize) - offset;
				ZeroMemory(pOutput, size * 2);
				pOutput += size;
			}

			offset = WINPR_ASSERTING_INT_CAST(size_t, (pOutput - pDstData));

			if ((DstSize < 0) || (offset != (size_t)DstSize))
				return -1;

			return 1;
		}
	}

} // namespace RLGR