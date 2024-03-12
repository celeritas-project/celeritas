//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ScintillationParams.cc
//---------------------------------------------------------------------------//
#include "ScintillationParams.hh"

#include <algorithm>
#include <numeric>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/math/SoftEqual.hh"
#include "celeritas/Types.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "ranges"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with imported data.
 */
std::shared_ptr<ScintillationParams>
ScintillationParams::from_import(ImportData const& data,
                                 SPConstParticles particle_params)
{
    CELER_EXPECT(!data.optical.empty());

    if (!std::any_of(
            data.optical.begin(), data.optical.end(), [](auto const& iter) {
                return static_cast<bool>(iter.second.scintillation);
            }))
    {
        // No scintillation data present
        return nullptr;
    }

    Input input;
    for (auto const& [matid, iom] : data.optical)
    {
        input.optmatid_to_matid.push_back(MaterialId(matid));
        input.data.push_back(iom.scintillation);

        if (!input.scintpid_to_pid.empty())
        {
            // Only store list of scintillation particle ids once
            continue;
        }

        auto const& iomsp = iom.scintillation.particles;
        for (auto pid : range(data.particles.size()))
        {
            auto imp_part_pdg = data.particles[pid].pdg;
            for (auto const& [pdg, ispc] : iomsp)
            {
                if (pdg == imp_part_pdg)
                {
                    input.scintpid_to_pid.push_back(ParticleId{pid});
                }
            }
        }
        CELER_ASSERT(input.scintpid_to_pid.size() == iomsp.size());
    }

    return std::make_shared<ScintillationParams>(std::move(input),
                                                 std::move(particle_params));
}

//---------------------------------------------------------------------------//
/*!
 * Construct with scintillation input data.
 */
ScintillationParams::ScintillationParams(Input const& input,
                                         SPConstParticles particle_params)
{
    CELER_EXPECT(input);
    HostVal<ScintillationData> host_data;

    auto const num_mat = input.optmatid_to_matid.size();
    auto const num_part = input.scintpid_to_pid.size();

    host_data.num_materials = num_mat;
    host_data.num_particles = num_part;

    CollectionBuilder scint_pid(&host_data.scint_particle_id);
    CollectionBuilder opt_mat_id(&host_data.optical_mat_id);
    CollectionBuilder materials(&host_data.materials);
    CollectionBuilder mat_components(&host_data.material_components);
    CollectionBuilder particles(&host_data.particles);
    CollectionBuilder part_components(&host_data.particle_components);

    // Store material ids
    std::vector<MaterialId> vec_optmatid(num_mat);
    for (auto matid : input.optmatid_to_matid)
    {
        opt_mat_id.push_back(MaterialId{matid});
    }

    // Store particle ids
    for (auto pid : input.scintpid_to_pid)
    {
        scint_pid.push_back(ParticleId{pid});
    }

    // Store material-only scintillation data
    for (auto const& inp : input.data)
    {
        // Check validity of input scintillation data
        CELER_ASSERT(inp);

        // Material-only data
        MaterialScintillationSpectrum mat_spec;
        mat_spec.yield = inp.material.yield;
        mat_spec.resolution_scale = inp.material.resolution_scale;
        auto comps = this->copy_components(inp.material.components);
        mat_spec.components
            = mat_components.insert_back(comps.begin(), comps.end());
        materials.push_back(mat_spec);
    }

    // Store particle- and material-dependent scintillation data
    // Index should loop over particles first, then materials. This ordering
    // should improve memory usage, since we group the material list for each
    // particle, and each particle (i.e. stream) traverses multiple materials
    for (auto pid : range(num_part))
    {
        auto const pdg = particle_params->id_to_pdg(ParticleId{pid}).get();
        for (auto matid : range(num_mat))
        {
            // Particle- and material-dependend scintillation spectrum
            auto iter = input.data[matid].particles.find(pdg);
            CELER_ASSERT(iter != input.data[matid].particles.end());
            CELER_ASSERT(pdg == iter->first);

            auto const& ispc = iter->second;
            ParticleScintillationSpectrum part_spec;
            // TODO: part_spec.yield_vector = this->copy(ipsc.yield_vector);
            auto comps = this->copy_components(ispc.components);
            part_spec.components
                = part_components.insert_back(comps.begin(), comps.end());
            particles.push_back(part_spec);
        }
    }
    CELER_ASSERT(particles.size() == num_part * num_mat);

    mirror_ = CollectionMirror<ScintillationData>{std::move(host_data)};
    CELER_ENSURE(mirror_);
}

//---------------------------------------------------------------------------//
/*
 * Return a Celeritas ScintillationComponent from an imported one.
 */
std::vector<ScintillationComponent> ScintillationParams::copy_components(
    std::vector<ImportScintComponent> const& input_comp)
{
    std::vector<ScintillationComponent> comp;
    real_type norm{0};
    for (auto i : range(comp.size()))
    {
        comp[i].lambda_mean = input_comp[i].lambda_mean;
        comp[i].lambda_sigma = input_comp[i].lambda_sigma;
        comp[i].rise_time = input_comp[i].rise_time;
        comp[i].fall_time = input_comp[i].fall_time;
        norm += input_comp[i].yield;
    }

    // Normalize yield
    for (auto i : range(comp.size()))
    {
        comp[i].yield_frac = input_comp[i].yield / norm;
    }

    this->validate(comp);
    return comp;
}

//---------------------------------------------------------------------------//
/*
 * Verify the correctness of the populated vector<ScintillationComponent>.
 */
void ScintillationParams::validate(
    std::vector<ScintillationComponent> const& vec_comp)
{
    for (auto i : range(vec_comp.size()))
    {
        CELER_VALIDATE(vec_comp[i].yield_frac > 0,
                       << "invalid yield_prob=" << vec_comp[i].yield_frac
                       << " for scintillation component " << i
                       << " (should be positive)");
        CELER_VALIDATE(vec_comp[i].lambda_mean > 0,
                       << "invalid lambda_mean=" << vec_comp[i].lambda_mean
                       << " for scintillation component " << i
                       << " (should be positive)");
        CELER_VALIDATE(vec_comp[i].lambda_sigma > 0,
                       << "invalid lambda_sigma=" << vec_comp[i].lambda_sigma
                       << " for scintillation component " << i
                       << " (should be positive)");
        CELER_VALIDATE(vec_comp[i].rise_time >= 0,
                       << "invalid rise_time=" << vec_comp[i].rise_time
                       << " for scintillation component " << i
                       << " (should be nonnegative)");
        CELER_VALIDATE(vec_comp[i].fall_time > 0,
                       << "invalid fall_time=" << vec_comp[i].fall_time
                       << " for scintillation component " << i
                       << " (should be positive)");
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
