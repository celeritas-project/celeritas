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
CutoffParams::CutoffParams(const Input& input)
{
    CELER_EXPECT(input.materials);
    CELER_EXPECT(input.particles);
    CELER_EXPECT(input.cutoffs.size() > 0);

    HostValue host_data;
    host_data.num_materials = input.materials->size();
    host_data.num_particles = input.particles->size();
    const auto cutoffs_size = host_data.num_materials * host_data.num_particles;

    auto cutoffs = make_builder(&host_data.cutoffs);
    cutoffs.reserve(cutoffs_size);

    for (const auto& per_material_cutoffs : input.cutoffs)
    {
        for (const auto& material_cutoffs : per_material_cutoffs.cutoffs)
        {
            cutoffs.push_back(std::move(material_cutoffs));
        }
    }

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<CutoffParamsData>{std::move(host_data)};
    CELER_ENSURE(this->host_pointers().cutoffs.size() == cutoffs_size);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
