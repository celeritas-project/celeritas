//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/mat/Material.test.hh
//---------------------------------------------------------------------------//

#include <vector>

#include "celeritas/mat/MaterialData.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Input data
struct MTestInput
{
    using MaterialParamsRef = DeviceCRef<MaterialParamsData>;
    using MaterialStateRef = DeviceRef<MaterialStateData>;

    MaterialParamsRef params;
    MaterialStateRef states;
    std::vector<MaterialTrackState> init;

    size_type size() const
    {
        CELER_EXPECT(states.size() == init.size());
        return states.size();
    }
};

//---------------------------------------------------------------------------//
//! Output results
struct MTestOutput
{
    std::vector<real_type> temperatures;
    std::vector<real_type> rad_len;
    std::vector<real_type> tot_z;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
MTestOutput m_test(MTestInput const& inp);

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
