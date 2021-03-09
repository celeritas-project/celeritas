//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CutoffParams.cc
//---------------------------------------------------------------------------//
#include "CutoffParams.hh"
#include "base/CollectionBuilder.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct on both host and device.
 */
CutoffParams::CutoffParams(Input& input)
{
    CELER_EXPECT(input.size() > 0);

    HostValue host_data;
    host_data.num_materials = input.size();
    host_data.num_particles = input.at(0).size();
    auto cutoffs_size = host_data.num_materials * host_data.num_particles;

    auto cutoffs = make_builder(&host_data.cutoffs);
    cutoffs.reserve(cutoffs_size);

    for (const auto& material_cutoffs : input)
    {
        for (const auto& particle_cutoff : material_cutoffs)
        {
            cutoffs.push_back(std::move(particle_cutoff));
        }
    }

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<CutoffParamsData>{std::move(host_data)};
    CELER_ENSURE(this->host_pointers().cutoffs.size() == cutoffs_size);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
