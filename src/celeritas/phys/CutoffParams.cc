//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
#include "corecel/sys/ScopedMem.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportMaterial.hh"
#include "celeritas/mat/MaterialParams.hh"

#include "CutoffData.hh"  // IWYU pragma: associated
#include "ParticleParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with imported data.
 */
std::shared_ptr<CutoffParams>
CutoffParams::from_import(ImportData const& data,
                          SPConstParticles particle_params,
                          SPConstMaterials material_params)
{
    CELER_EXPECT(!data.phys_materials.empty());
    CELER_EXPECT(particle_params);
    CELER_EXPECT(material_params);

    CutoffParams::Input input;
    input.particles = std::move(particle_params);
    input.materials = std::move(material_params);

    for (auto const& pdg : CutoffParams::pdg_numbers())
    {
        CutoffParams::MaterialCutoffs mat_cutoffs;
        for (auto const& material : data.phys_materials)
        {
            auto iter = material.pdg_cutoffs.find(pdg.get());
            if (iter != material.pdg_cutoffs.end())
            {
                // Found assigned cutoff values
                mat_cutoffs.push_back(
                    {units::MevEnergy(iter->second.energy),
                     static_cast<real_type>(iter->second.range)});
            }
            else
            {
                mat_cutoffs.push_back({zero_quantity(), 0});
            }
        }
        input.cutoffs.insert({pdg, mat_cutoffs});
    }
    input.apply_post_interaction = data.em_params.apply_cuts;

    return std::make_shared<CutoffParams>(input);
}

//---------------------------------------------------------------------------//
/*!
 * Construct on both host and device.
 */
CutoffParams::CutoffParams(Input const& input)
{
    CELER_EXPECT(input.materials);
    CELER_EXPECT(input.particles);

    ScopedMem record_mem("CutoffParams.construct");

    HostValue host_data;
    host_data.num_materials = input.materials->size();
    host_data.apply_post_interaction = input.apply_post_interaction;
    if (input.apply_post_interaction)
    {
        host_data.ids.electron = input.particles->find(pdg::electron());
        host_data.ids.positron = input.particles->find(pdg::positron());
        host_data.ids.gamma = input.particles->find(pdg::gamma());
    }

    std::vector<ParticleCutoff> cutoffs;

    // Initialize mapping of particle ID to index with invalid indices
    std::vector<size_type> id_to_index(input.particles->size(), size_type(-1));
    size_type current_index = 0;

    for (auto const& pdg : CutoffParams::pdg_numbers())
    {
        if (auto pid = input.particles->find(pdg))
        {
            id_to_index[pid.get()] = current_index++;

            auto iter = input.cutoffs.find(pdg);
            if (iter != input.cutoffs.end())
            {
                // Found valid PDG and cutoff values
                auto const& mat_cutoffs = iter->second;
                CELER_ASSERT(mat_cutoffs.size() == host_data.num_materials);
                cutoffs.insert(
                    cutoffs.end(), mat_cutoffs.begin(), mat_cutoffs.end());
            }
            else
            {
                // Particle was defined in the problem but does not have
                // cutoffs assigned -- set cutoffs to zero
                for ([[maybe_unused]] auto i : range(host_data.num_materials))
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
    data_ = ParamsDataStore<CutoffParamsData>{std::move(host_data)};
    CELER_ENSURE(data_);
}

//---------------------------------------------------------------------------//
/*!
 * PDG numbers of particles with prodution cuts.
 *
 * Positron production cuts are only used when the \c apply_post_interaction
 * option is enabled to explicitly kill secondary positrons with energies below
 * the production threshold. Proton production cuts are not currently used.
 */
std::vector<PDGNumber> const& CutoffParams::pdg_numbers()
{
    static std::vector<PDGNumber> const pdg_numbers{
        pdg::electron(), pdg::gamma(), pdg::positron()};
    return pdg_numbers;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
