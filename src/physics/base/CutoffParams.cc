//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file CutoffParams.cc
//---------------------------------------------------------------------------//
#include "CutoffParams.hh"
#include "base/CollectionBuilder.hh"
#include "physics/base/Units.hh"

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
    CELER_EXPECT(!input.cutoffs.empty());

    HostValue host_data;
    host_data.num_materials = input.materials->size();
    host_data.num_particles = input.particles->size();
    const auto cutoffs_size = host_data.num_materials * host_data.num_particles;

    auto host_cutoffs = make_builder(&host_data.cutoffs);
    host_cutoffs.reserve(cutoffs_size);

    for (const auto pid : range(ParticleId{input.particles->size()}))
    {
        const auto  pdg  = input.particles->id_to_pdg(pid);
        const auto& iter = input.cutoffs.find(pdg);

        if (iter != input.cutoffs.end())
        {
            // Found valid PDG and cutoff values
            const auto& vec_mat_cutoffs = iter->second;
            CELER_ASSERT(vec_mat_cutoffs.size() == host_data.num_materials);
            host_cutoffs.insert_back(vec_mat_cutoffs.begin(),
                                     vec_mat_cutoffs.end());
        }
        else
        {
            // PDG not added to Input.cutoffs. Set cutoffs to zero
            for (int i : range(host_data.num_materials))
            {
                host_cutoffs.push_back({units::MevEnergy{zero_quantity()}, 0});
            }
        }
    }

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<CutoffParamsData>{std::move(host_data)};
    CELER_ENSURE(this->host_pointers().cutoffs.size() == cutoffs_size);
}

//---------------------------------------------------------------------------//
} // namespace celeritas
