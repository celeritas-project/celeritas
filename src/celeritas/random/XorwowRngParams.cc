//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/random/XorwowRngParams.cc
//---------------------------------------------------------------------------//
#include "XorwowRngParams.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/cont/Array.hh"
#include "celeritas/random/XorwowRngData.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a low-entropy seed.
 */
XorwowRngParams::XorwowRngParams(unsigned int seed)
{
    HostVal<XorwowRngParamsData> host_data;
    host_data.seed = {seed};
    host_data.jump = this->get_jump_poly();
    host_data.jump_subsequence = this->get_jump_subsequence_poly();
    CELER_ASSERT(host_data);
    data_ = CollectionMirror<XorwowRngParamsData>{std::move(host_data)};
}

//---------------------------------------------------------------------------//
/*!
 * Get the jump polynomials.
 *
 * The jump polynomials (and the jump matrices) can be generated using
 * https://github.com/celeritas-project/utils/blob/main/prng/xorwow-jump.py.
 *
 * The coefficients of the polynomial are packed into a 160-bit integer, which
 * is then split into 5 32-bit integers, ordered from the lower 32 bits to the
 * upper 32 bits.
 *
 * The jump sizes are 4^i for i = [0, 10).
 */
auto XorwowRngParams::get_jump_poly() -> ArrayJumpPoly const&
{
    static ArrayJumpPoly const jump = {
        {{0x00000002u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
         {0x00000010u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
         {0x00010000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
         {0x00000000u, 0x00000000u, 0x00000001u, 0x00000000u, 0x00000000u},
         {0xbebd3534u, 0x7064f5bcu, 0x20be29ebu, 0x536d5b32u, 0x063a0069u},
         {0xed64ec08u, 0xafc48684u, 0xd81c59eeu, 0x1640314fu, 0x2bf0ccefu},
         {0x9d10e028u, 0xee56d79cu, 0xfb1b3286u, 0x0f747418u, 0x26f9476du},
         {0x3f490634u, 0x9ae593fcu, 0x1a95bb6bu, 0xda10a3fcu, 0xa3abaf54u},
         {0xfb9680e9u, 0xbdaba0b2u, 0x3986540fu, 0x23fe6cccu, 0x0994e82fu},
         {0x32da6db4u, 0x80135829u, 0x3abd4734u, 0x2060c3f9u, 0x38b2dd97u}}};
    return jump;
}

//---------------------------------------------------------------------------//
/*!
 * Get the jump polynomials for jumping over subsequences.
 *
 * A subsequence is 2^67 steps. The jump sizes are 4^i * 2^67 for i = [0, 10).
 */
auto XorwowRngParams::get_jump_subsequence_poly() -> ArrayJumpPoly const&
{
    static ArrayJumpPoly const jump_subsequence = {
        {{0x26294934u, 0x77bbc248u, 0x1a87dad0u, 0x930052d4u, 0x947e6dd2u},
         {0xa7474d19u, 0x37c549e0u, 0x140877d2u, 0x24c43924u, 0xcd52ebecu},
         {0xfcc8f692u, 0x35aa698au, 0x04ebbf85u, 0x448304b0u, 0x082f3fd5u},
         {0xe502f5b3u, 0x77859d31u, 0x97e1cbb3u, 0xea09047au, 0x61d5f37eu},
         {0xb3a15db4u, 0xc6df3330u, 0x1a8be751u, 0xf1e4b221u, 0x6bb61c05u},
         {0x19e2bee8u, 0xf8974218u, 0x4e65536au, 0xa2570336u, 0xe9b88082u},
         {0x0e2cae2eu, 0xd1011279u, 0x58923768u, 0x2bf650bau, 0xc985bcdau},
         {0x146281e7u, 0xa45b4452u, 0xafa8c695u, 0x74a0ac94u, 0x4b250e0au},
         {0x89658c8bu, 0xf3914315u, 0xa73fe84bu, 0x3c5fadb6u, 0xadba8dd6u},
         {0x82161afcu, 0x9bb13c55u, 0xfd20d7fbu, 0x306d90b9u, 0x92bf386fu}}};
    return jump_subsequence;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
