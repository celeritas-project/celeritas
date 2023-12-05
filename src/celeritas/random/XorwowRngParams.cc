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
 * The jump polynomials (as well as the jump matrices) are calculated using
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
    static ArrayJumpPoly const jump
        = {{{0x2, 0x0, 0x0, 0x0, 0x0},
            {0x10, 0x0, 0x0, 0x0, 0x0},
            {0x10000, 0x0, 0x0, 0x0, 0x0},
            {0x0, 0x0, 0x1, 0x0, 0x0},
            {0xbebd3534, 0x7064f5bc, 0x20be29eb, 0x536d5b32, 0x63a0069},
            {0xed64ec08, 0xafc48684, 0xd81c59ee, 0x1640314f, 0x2bf0ccef},
            {0x9d10e028, 0xee56d79c, 0xfb1b3286, 0xf747418, 0x26f9476d},
            {0x3f490634, 0x9ae593fc, 0x1a95bb6b, 0xda10a3fc, 0xa3abaf54},
            {0xfb9680e9, 0xbdaba0b2, 0x3986540f, 0x23fe6ccc, 0x994e82f},
            {0x32da6db4, 0x80135829, 0x3abd4734, 0x2060c3f9, 0x38b2dd97}}};
    return jump;
}

//---------------------------------------------------------------------------//
/*!
 * Get the jump polynomials for jumping over subsequences.
 *
 * The jump sizes are 4^i * 2^67 for i = [0, 10).
 */
auto XorwowRngParams::get_jump_subsequence_poly() -> ArrayJumpPoly const&
{
    static ArrayJumpPoly const jump_subsequence
        = {{{0x26294934, 0x77bbc248, 0x1a87dad0, 0x930052d4, 0x947e6dd2},
            {0xa7474d19, 0x37c549e0, 0x140877d2, 0x24c43924, 0xcd52ebec},
            {0xfcc8f692, 0x35aa698a, 0x4ebbf85, 0x448304b0, 0x82f3fd5},
            {0xe502f5b3, 0x77859d31, 0x97e1cbb3, 0xea09047a, 0x61d5f37e},
            {0xb3a15db4, 0xc6df3330, 0x1a8be751, 0xf1e4b221, 0x6bb61c05},
            {0x19e2bee8, 0xf8974218, 0x4e65536a, 0xa2570336, 0xe9b88082},
            {0xe2cae2e, 0xd1011279, 0x58923768, 0x2bf650ba, 0xc985bcda},
            {0x146281e7, 0xa45b4452, 0xafa8c695, 0x74a0ac94, 0x4b250e0a},
            {0x89658c8b, 0xf3914315, 0xa73fe84b, 0x3c5fadb6, 0xadba8dd6},
            {0x82161afc, 0x9bb13c55, 0xfd20d7fb, 0x306d90b9, 0x92bf386f}}};
    return jump_subsequence;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
