//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/phys/CutoffParams.cc
//---------------------------------------------------------------------------//
#include "CutoffParams.hh"

#include <type_traits>
#include <utility>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportMaterial.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "CutoffData.hh" // IWYU pragma: associated

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

    for (const auto& pdg : CutoffParams::pdg_numbers())
    {
        CutoffParams::MaterialCutoffs mat_cutoffs;
        for (const auto& material : data.materials)
        {
            auto iter = material.pdg_cutoffs.find(pdg.get());
            if (iter != material.pdg_cutoffs.end())
            {
                // Found assigned cutoff values
                mat_cutoffs.push_back({units::MevEnergy{iter->second.energy},
                                       iter->second.range});
            }
            else
            {
                mat_cutoffs.push_back({zero_quantity(), 0});
            }
        }
        input.cutoffs.insert({pdg, mat_cutoffs});
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

    std::vector<ParticleCutoff> cutoffs;

    // Initialize mapping of particle ID to index with invalid indices
    std::vector<size_type> id_to_index(input.particles->size(), size_type(-1));
    size_type              current_index = 0;

    for (const auto& pdg : CutoffParams::pdg_numbers())
    {
        if (auto pid = input.particles->find(pdg))
        {
            id_to_index[pid.get()] = current_index++;

            auto iter = input.cutoffs.find(pdg);
            if (iter != input.cutoffs.end())
            {
                // Found valid PDG and cutoff values
                const auto& mat_cutoffs = iter->second;
                CELER_ASSERT(mat_cutoffs.size() == host_data.num_materials);
                cutoffs.insert(
                    cutoffs.end(), mat_cutoffs.begin(), mat_cutoffs.end());
            }
            else
            {
                // Particle was defined in the problem but does not have
                // cutoffs assigned -- set cutoffs to zero
                for (CELER_MAYBE_UNUSED auto i : range(host_data.num_materials))
                {
                    cutoffs.push_back({zero_quantity(), 0});
                }
            }
        }
    }
    CELER_ASSERT(current_index <= CutoffParams::pdg_numbers().size());
    host_data.num_particles = current_index;
    make_builder(&host_data.cutoffs).insert_back(cutoffs.begin(), cutoffs.end());
    make_builder(&host_data.id_to_index)
        .insert_back(id_to_index.begin(), id_to_index.end());

    // Move to mirrored data, copying to device
    data_ = CollectionMirror<CutoffParamsData>{std::move(host_data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * PDG numbers of particles with prodution cuts.
 *
 * Only electrons and photons have secondary production cuts -- protons are not
 * currently used, and positrons cannot have production cuts.
 */
const std::vector<PDGNumber>& CutoffParams::pdg_numbers()
{
    static const std::vector<PDGNumber> pdg_numbers{pdg::electron(),
                                                    pdg::gamma()};
    return pdg_numbers;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
