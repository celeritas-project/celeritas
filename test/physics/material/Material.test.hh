//----------------------------------*-C++-*----------------------------------//
// Copyright 2020-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file physics/material/Material.test.hh
//---------------------------------------------------------------------------//

#include <vector>

#include "celeritas/mat/MaterialData.hh"

namespace celeritas_test
{
using namespace celeritas;
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Input data
struct MTestInput
{
    using MaterialParamsRef
        = MaterialParamsData<Ownership::const_reference, MemSpace::device>;
    using MaterialStateRef
        = MaterialStateData<Ownership::reference, MemSpace::device>;

    MaterialParamsRef               params;
    MaterialStateRef                states;
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
MTestOutput m_test(const MTestInput& inp);

//---------------------------------------------------------------------------//
} // namespace celeritas_test
