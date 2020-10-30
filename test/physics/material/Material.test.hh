//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Material.test.hh
//---------------------------------------------------------------------------//

#include <vector>
#include "physics/material/MaterialParamsPointers.hh"
#include "physics/material/MaterialStatePointers.hh"

namespace celeritas_test
{
using namespace celeritas;
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Input data
struct MTestInput
{
    MaterialParamsPointers          params;
    MaterialStatePointers           states;
    std::vector<MaterialTrackState> init;

    size_type size() const
    {
        REQUIRE(states.size() == init.size());
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
MTestOutput m_test(const MTestInput& inp);

//---------------------------------------------------------------------------//
} // namespace celeritas_test
