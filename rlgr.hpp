#pragma once
#include <cstdint>
#include <algorithm>
#include <stdexcept>
namespace RLGR
{

	/**
	 * FreeRDP: A Remote Desktop Protocol Implementation
	 * RemoteFX Codec Library - RLGR
	 *
	 * Copyright 2011 Vic Lee
	 *
	 * Licensed under the Apache License, Version 2.0 (the "License");
	 * you may not use this file except in compliance with the License.
	 * You may obtain a copy of the License at
	 *
	 *     http://www.apache.org/licenses/LICENSE-2.0
	 *
	 * Unless required by applicable law or agreed to in writing, software
	 * distributed under the License is distributed on an "AS IS" BASIS,
	 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	 * See the License for the specific language governing permissions and
	 * limitations under the License.
	 */

	/**
	 * This implementation of RLGR refers to
	 * [MS-RDPRFX] 3.1.8.1.7.3 RLGR1/RLGR3 Pseudocode
	 */

	/* Constants used in RLGR1/RLGR3 algorithm */
	constexpr uint32_t KPMAX = 80; /* max value for kp or krp */
	constexpr uint32_t LSGR = 3;   /* shift count to convert kp to k */
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

	inline uint32_t lzcnt_s(uint32_t x)
	{
		return (x) ? __builtin_clz(x) : 32;
	}

	template <typename BitReader>
	struct Decoder
	{
		BitReader bitreader;
		uint32_t k = 1;
		uint32_t kp = 1 << LSGR;
		uint32_t kr = 1;
		uint32_t krp = 1 << LSGR;
		Decoder(BitReader bitreader) : bitreader(bitreader) {}

		void init() {
			bitreader.init();
		}

		std::pair<int, size_t> decode()
		{

			int mag{};
			uint32_t code{0};
			uint32_t cnt{};
			uint32_t vk{};
			if (k)
			{
				/* Run-Length (RL) Mode */

				size_t run = 0;

				/* count number of leading 0s */

				cnt = lzcnt_s(bitreader.peek32());

				vk = (cnt);

				while (cnt == 32)
				{
					bitreader.skip(32);
					cnt = lzcnt_s(bitreader.peek32());
					vk += cnt;
				}

				bitreader.skip((vk % 32) + 1);

				while (vk--)
				{
					const uint32_t add = (1 << k); /* add (1 << k) to run length */
					run += add;

					/* update k, kp params */

					kp += UP_GR;

					if (kp > KPMAX)
						kp = KPMAX;

					k = kp >> LSGR;
				}

				/* next k bits contain run length remainder */

				run += bitreader.peek32() >> (32 - k);
				bitreader.skip(k);

				/* read sign bit */

				int sign = (bitreader.peek32() & 0x80000000U) ? 1 : 0;
				bitreader.skip(1);

				/* count number of leading 1s */

				cnt = lzcnt_s(~bitreader.peek32());

				vk = cnt;

				while (cnt == 32)
				{
					bitreader.skip(32);

					cnt = lzcnt_s(~(bitreader.peek32()));

					vk += cnt;
				}

				bitreader.skip((vk % 32) + 1);

				/* next kr bits contain code remainder */

				if (kr > 0)
				{
					code = (bitreader.peek32() >> (32 - kr));
					bitreader.skip(kr);
				}

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
					mag = static_cast<int>(code + 1) * -1;
				else
					mag = static_cast<int>(code + 1);

				/* write to output stream */

				return {mag, run};
			}
			else
			{
				/* Golomb-Rice (GR) Mode */

				/* count number of leading 1s */

				cnt = lzcnt_s(~(bitreader.peek32()));
				vk = cnt;

				while (cnt == 32)
				{
					bitreader.skip(32);
					cnt = lzcnt_s(~(bitreader.peek32()));
					vk += cnt;
				}

				bitreader.skip((vk % 32) + 1);

				/* next kr bits contain code remainder */

				if (kr > 0)
				{
					code = (bitreader.peek32() >> (32 - kr));
				}
				bitreader.skip(kr);

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

				/* RLGR1 */
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
						mag = static_cast<int>((code + 1) >> 1) * -1;
					else
						mag = static_cast<int>(code >> 1);
				}

				return {mag, size_t{0}};
			}
		}
		std::pair<int, size_t> rle_state{0, static_cast<size_t>(-1)};
		int decode_rle() {
			if (rle_state.second == static_cast<size_t>(-1)) {
				rle_state = decode();
			}
			int res = (rle_state.second == 0 ? rle_state.first : 0);
			rle_state.second -= 1;
			return res;
		}
	};

	template <typename BitWriter>
	struct Encoder
	{
		/* Emit bitPattern to the output bitstream */
		BitWriter bitwriter;
		uint32_t k = 1;
		uint32_t kp = 1 << LSGR;
		uint32_t krp = 1 << LSGR;
		size_t numZeros{0};

		Encoder(BitWriter bitwriter) : bitwriter(bitwriter) {}

		/* Converts the input value to (2 * abs(input) - sign(input)), where sign(input) = (input < 0 ? 1 :
		 * 0) and returns it */
		inline uint32_t Get2MagSign(int32_t input)
		{
			if (input >= 0)
				return static_cast<uint32_t>(2 * input);
			return static_cast<uint32_t>(-2 * input - 1);
		}

		/* Outputs the Golomb/Rice encoding of a non-negative integer */
		void CodeGR(uint32_t &krp, uint32_t val)
		{
			uint32_t kr = krp >> LSGR;

			/* unary part of GR code */

			const uint32_t vk = val >> kr;
			// Original: OutputBit(bs, vk, 1) -> sequence of bits
			// Original: OutputBit(bs, 1, 0) -> sequence of bits

			auto quotient = vk;
			while (quotient >= 32)
			{
				bitwriter.put_bits(32, 0xffffffffu);
				quotient -= 32;
			}
			bitwriter.put_bits(quotient + 1, ((1ULL << (quotient)) - 1) << 1); // Write quotient bits

			/* remainder part of GR code, if needed */
			if (kr)
			{
				bitwriter.put_bits(kr, val & ((1 << kr) - 1)); // Original: OutputBits(kr, ...) -> pattern
			}

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

		/* process all the input coefficients */
		void encode(int input)
		{

			if (k)
			{
				/* collect the run of zeros in the input stream */
				/* RUN-LENGTH MODE */
				if (input == 0)
				{
					numZeros++;
					return;
				}

				// emit output zeros
				uint32_t runmax = 1 << k;
				while (numZeros >= runmax)
				{
					bitwriter.put_bits(1, 0); /* output a zero bit */ // Original: OutputBit(bs, 1, 0) -> sequence of bits
					numZeros -= runmax;
					k = UpdateParam(kp, UP_GR); /* update kp, k */
					runmax = 1 << k;
				}

				/* output a 1 to terminate runs */
				bitwriter.put_bits(1, 1); // Original: OutputBit(bs, 1, 1) -> sequence of bits

				/* output the remaining run length using k bits */
				bitwriter.put_bits(k, numZeros); // Original: OutputBits(k, numZeros) -> pattern
				numZeros = 0;

				/* note: when we reach here and the last byte being encoded is 0, we still
				   need to output the last two bits, otherwise mstsc will crash */

				/* encode the nonzero value using GR coding */
				const uint32_t mag = static_cast<uint32_t>(input < 0 ? -input : input); /* absolute value of input coefficient */
				/* sign of input coefficient */
				bitwriter.put_bits(1, (input < 0 ? 1 : 0)); /* output the sign bit */ // Original: OutputBits(1, sign) -> pattern
				CodeGR(krp, mag ? mag - 1 : 0);								  /* output GR code for (mag - 1) */

				k = UpdateParam(kp, -DN_GR);
			}
			else
			{
				/* GOLOMB-RICE MODE */

				/* convert input to (2*magnitude - sign), encode using GR code */
				uint32_t twoMs = Get2MagSign(input);
				CodeGR(krp, twoMs);

				/* update k, kp */
				/* NOTE: as of Aug 2011, the algorithm is still wrongly documented
				   and the update direction is reversed */
				if (twoMs)
				{
					k = UpdateParam(kp, -DQ_GR);
				}
				else
				{
					k = UpdateParam(kp, UQ_GR);
				}
			}
		}
		void flush() {
			encode(-1);
			bitwriter.flush();
		}
	};

} // namespace RLGR