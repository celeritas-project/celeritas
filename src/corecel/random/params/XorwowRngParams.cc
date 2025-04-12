//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/random/params/XorwowRngParams.cc
//---------------------------------------------------------------------------//
#include "XorwowRngParams.hh"

#include <utility>

#include "corecel/Assert.hh"
#include "corecel/cont/Array.hh"
#include "corecel/random/data/XorwowRngData.hh"

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
 * The jump sizes are 4^i for i = [0, 32).
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
         {0x32da6db4u, 0x80135829u, 0x3abd4734u, 0x2060c3f9u, 0x38b2dd97u},
         {0x0c5b636fu, 0x4407a814u, 0x60204515u, 0x5bd4dbd2u, 0x2509eeb5u},
         {0x21e9179du, 0x2b57aa94u, 0x5f06e1fcu, 0x6735dc98u, 0x9aa0181fu},
         {0x793adf2bu, 0x3d3e75c8u, 0x0b091758u, 0x9deb3f50u, 0xbcd116ecu},
         {0xfa8fc346u, 0x694921b1u, 0xf2bd4c48u, 0x8b05ae1bu, 0x2de7aee2u},
         {0xf144e1f3u, 0xb9b4b3c4u, 0x8222f622u, 0x9072105au, 0x66083550u},
         {0xbd734cf1u, 0x9254a905u, 0xfb38236du, 0x11e62fd3u, 0xf9e7d21eu},
         {0x943999e4u, 0xc05db913u, 0x4e4010f3u, 0x9b865d3du, 0x0fd64174u},
         {0x19eb1bbbu, 0xbaea3750u, 0x0fa8f206u, 0xd49dd019u, 0x9b3bccafu},
         {0xc97b5642u, 0x2ebac13du, 0xbfe04058u, 0x2c6a7132u, 0x576780a5u},
         {0x0ac5eea9u, 0xa37bfcd3u, 0x790ec91du, 0xba339dbcu, 0xc83db5cdu},
         {0xa33b53ffu, 0x1ce9360du, 0x4727f89bu, 0x05eacbcdu, 0x01632278u},
         {0x22b4f98cu, 0xd23a7f5au, 0x8d420eddu, 0xeadda806u, 0xcfd2a002u},
         {0xf66ad52bu, 0x3ab8e3d7u, 0x1e8352a4u, 0xe44a8605u, 0x6c106869u},
         {0x79a31c08u, 0xc28d5d18u, 0x91649708u, 0x8b1bb8f8u, 0xf9158a86u},
         {0x670f870eu, 0xa7bb9766u, 0xef013c78u, 0xeb4a1373u, 0x256f3323u},
         {0xf333af18u, 0x3b4e266bu, 0xa872663eu, 0x3888cd82u, 0x4daf13ecu},
         {0x75689dc9u, 0x036bf3a9u, 0x64ce979cu, 0x3bdff14fu, 0x51e43048u},
         {0xef06fe75u, 0xefd6d1d8u, 0x09319075u, 0x69d568f5u, 0x5b2bd898u},
         {0xf05ae255u, 0xc4df4ca0u, 0xf032420cu, 0xf44ae9f0u, 0xe0298de2u},
         {0x02308bc9u, 0xdbe74deeu, 0xf4c5fe7du, 0xaac571a1u, 0xaa1f5f8bu},
         {0x6e8043cau, 0x0ed24ff6u, 0x1e668b6au, 0x538fc45fu, 0x4bfb509du},
         {0x9f564f7bu, 0x543973e4u, 0x9b33ee2cu, 0x149df73au, 0x58f31585u}}};
    return jump;
}

//---------------------------------------------------------------------------//
/*!
 * Get the jump polynomials for jumping over subsequences.
 *
 * A subsequence is 2^67 steps. The jump sizes are 4^i * 2^67 for i = [0, 32).
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
         {0x82161afcu, 0x9bb13c55u, 0xfd20d7fbu, 0x306d90b9u, 0x92bf386fu},
         {0x3e9058c1u, 0x71da9705u, 0xe2cb1f2bu, 0x73456536u, 0xbd6501b6u},
         {0xb1321eb5u, 0xb06d01a2u, 0x51532012u, 0x8fb59962u, 0x141d3b0bu},
         {0xa0ffe9a4u, 0xf57be00eu, 0xe706880cu, 0x191211efu, 0xee5664fbu},
         {0xe129d45du, 0xac3e698bu, 0xd61f79fdu, 0x7a72a28bu, 0x7f4942beu},
         {0x874e26a7u, 0x33bc5a47u, 0xad95cba5u, 0x67651c39u, 0xf1f07dcau},
         {0x4b324c2eu, 0x2bdcc5a0u, 0x53eb4240u, 0xcac4fbc1u, 0x13e529e7u},
         {0x5fe4b704u, 0xd77445c3u, 0xb80eeb3cu, 0x6720fc6du, 0x7da33f71u},
         {0x27786b0du, 0x55a8b8bbu, 0xad73d087u, 0x548172a6u, 0xb8dcb607u},
         {0x1e9372ddu, 0xe081adccu, 0xf9650df2u, 0x0ad599e4u, 0x21aeba83u},
         {0x552ec26fu, 0x2663dad8u, 0x25bf8d5eu, 0x538f9e9bu, 0x804bfd4cu},
         {0xab750c90u, 0x454415efu, 0xd94a347cu, 0xd23c81beu, 0x551c7096u},
         {0xbc6d2665u, 0xc1fd8153u, 0xd43c38c9u, 0xd70344ccu, 0x279357c0u},
         {0x88aced61u, 0x3925e5e8u, 0xc8af3caau, 0xefa299b1u, 0xbc1538f8u},
         {0xc3051a0au, 0x11a68894u, 0x0ec32c75u, 0xb9e1af76u, 0x45d20f13u},
         {0x54f062f0u, 0xcf7989d8u, 0x443e496bu, 0x17d83e81u, 0xa2be8639u},
         {0x267af43cu, 0x14dfd913u, 0x2dbb25b6u, 0x227b1a06u, 0x24402bc6u},
         {0xad66cdfeu, 0x7cfa6a50u, 0x8fca746au, 0x7b18f04bu, 0x28eeb28fu},
         {0x37abb017u, 0x7735fb03u, 0x557fb7cdu, 0x520ea993u, 0x69e4a18du},
         {0x53140fddu, 0x0dfb37a9u, 0x88772b05u, 0x20be07e3u, 0x128c07bdu},
         {0x1e72e926u, 0x829ca1d2u, 0x084c2bd7u, 0xcea065e7u, 0xd2b401bfu},
         {0x93d21898u, 0x97c7960eu, 0xb2899f9du, 0xd528a53du, 0x04f33fcau},
         {0x06e1a24fu, 0x4a7295afu, 0x5534bfe6u, 0x452ec8f1u, 0x79685920u}}};
    return jump_subsequence;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
