//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CutoffParams.cc
//---------------------------------------------------------------------------//
#include "CutoffParams.hh"

#include "base/PieBuilder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
CutoffParams::CutoffParams(Input& inp)
{
    CELER_EXPECT(inp.size() > 0);

    HostValue host_data;
    for (const auto& material_cutoffs : inp)
    {
        for (const auto& element_cutoff : material_cutoffs)
        {
            make_pie_builder(&host_data.cutoffs).push_back(element_cutoff);
        }
    }

    // Move to mirrored data, copying to device
    data_ = PieMirror<CutoffParamsData>{std::move(host_data)};
}

//---------------------------------------------------------------------------//
} // namespace celeritas
