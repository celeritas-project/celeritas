//------------------------------- -*- C++ -*- -------------------------------//
// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0
//---------------------------------------------------------------------------//
/*!
 * \file corecel/random/params/RanluxppRngParams.cc
 *
 * Original source:
 * https://github.com/apt-sim/AdePT/blob/master/include/AdePT/copcore/Ranluxpp.h
 */
//---------------------------------------------------------------------------//
#include "RanluxppRngParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a seed.
 */
RanluxppRngParams::RanluxppRngParams(RanluxppUInt seed)
{
    HostVal<RanluxppRngParamsData> host_data;
    host_data.seed = seed;
    host_data.kA_2048 = this->getKa();
    host_data.kMaxPos = 9 * 64;
    CELER_ASSERT(host_data);

    data_ = CollectionMirror<RanluxppRngParamsData>(std::move(host_data));
}

//---------------------------------------------------------------------------//
/*!
 * Get the Ka polynomial.
 */
RanluxppArray9 const& RanluxppRngParams::getKa() const
{
    static RanluxppArray9 const k_array = {
        0xed7faa90747aaad9,
        0x4cec2c78af55c101,
        0xe64dcb31c48228ec,
        0x6d8a15a13bee7cb0,
        0x20b2ca60cb78c509,
        0x256c3d3c662ea36c,
        0xff74e54107684ed2,
        0x492edfcc0cc8e753,
        0xb48c187cf5b22097,
    };
    return k_array;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
