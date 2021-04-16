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
#include "io/ImportData.hh"
#include "ParticleParams.hh"
#include "physics/material/MaterialParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with imported data.
 */
std::shared_ptr<CutoffParams>
CutoffParams::from_import(const ImportData& data,
                          SPConstParticles  particle_params,
                          SPConstMaterials  material_params)
{
    CELER_EXPECT(data);
    CELER_EXPECT(particle_params);
    CELER_EXPECT(material_params);

    CutoffParams::Input input;
    input.particles = std::move(particle_params);
    input.materials = std::move(material_params);

    for (const auto pid : range(ParticleId{input.particles->size()}))
    {
        CutoffParams::MaterialCutoffs m_cutoffs;

        const auto pdg = input.particles->id_to_pdg(pid);

        for (const auto& material : data.materials)
        {
            const auto& iter = material.pdg_cutoffs.find(pdg.get());

            ParticleCutoff p_cutoff;
            if (iter != material.pdg_cutoffs.end())
            {
                // Is a particle type with assigned cutoff values
                p_cutoff.energy = units::MevEnergy{iter->second.energy};
                p_cutoff.range  = iter->second.range;
            }
            else
            {
                // Set cutoffs to zero
                p_cutoff.energy = units::MevEnergy{zero_quantity()};
                p_cutoff.range  = 0;
            }
            m_cutoffs.push_back(p_cutoff);
        }
        input.cutoffs.insert({pdg, m_cutoffs});
    }

    return std::make_shared<CutoffParams>(input);
}

//---------------------------------------------------------------------------//
/*!
 * Construct on both host and device.
 */
CutoffParams::CutoffParams(const Input& input)
{
    CELER_EXPECT(input.materials);
    CELER_EXPECT(input.particles);

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
            for (CELER_MAYBE_UNUSED auto i : range(host_data.num_materials))
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
