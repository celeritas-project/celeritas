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
#include "corecel/data/DedupeCollectionBuilder.hh"
#include "corecel/math/SoftEqual.hh"
#include "celeritas/Types.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/phys/ParticleParams.hh"

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
    input.scintillation_by_particle
        = data.optical_params.scintillation_by_particle;
    input.matid_to_optmatid.resize(data.materials.size());
    input.pid_to_scintpid.resize(data.particles.size());

    OpticalMaterialId optmatid;
    ScintillationParticleId scintpid;
    for (auto const& [matid, iom] : data.optical)
    {
        input.matid_to_optmatid[matid] = optmatid++;
        input.data.push_back(iom.scintillation);

        if (!input.pid_to_scintpid.empty())
        {
            // Every material must have same number of scintillation particles
            CELER_ASSERT(input.pid_to_scintpid.size()
                         == iom.scintillation.particles.size());
            // Only store list of scintillation particle ids once
            continue;
        }

        // Loop over particles and add all that are scintillation particles
        auto const& iomsp = iom.scintillation.particles;
        for (auto pid : range(data.particles.size()))
        {
            if (iomsp.find(data.particles[pid].pdg) != iomsp.end())
            {
                // Add new ScintillationParticleId
                input.pid_to_scintpid[pid] = scintpid++;
            }
        }
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
    CollectionBuilder build_optmatid(&host_data.matid_to_optmatid);
    CollectionBuilder build_scintpid(&host_data.pid_to_scintpid);
    CollectionBuilder build_resolutionscale(&host_data.resolution_scale);
    CollectionBuilder build_compoments(&host_data.components);

    auto const num_mat = input.matid_to_optmatid.size();
    auto const num_part = input.pid_to_scintpid.size();

    // Store material ids
    for (auto const id : input.matid_to_optmatid)
    {
        build_optmatid.push_back(id);
        if (id)
        {
            host_data.num_opt_materials++;
        }
    }
    CELER_ENSURE(host_data.num_opt_materials > 0);

    // Store particle ids
    for (auto const id : input.pid_to_scintpid)
    {
        build_scintpid.push_back(id);
        if (id)
        {
            host_data.num_opt_particles++;
        }
    }
    CELER_ENSURE(host_data.num_opt_particles > 0);

    // Store resolution scale
    for (auto const& inp : input.data)
    {
        CELER_VALIDATE(inp.resolution_scale >= 0,
                       << "invalid resolution_scale=" << inp.resolution_scale
                       << " for scintillation (should be nonnegative)");
        build_resolutionscale.push_back(inp.resolution_scale);
    }
    CELER_ENSURE(build_resolutionscale.size() == num_mat);

    if (!input.scintillation_by_particle)
    {
        CollectionBuilder build_materials(&host_data.materials);

        // Store material scintillation data
        for (auto const& inp : input.data)
        {
            // Check validity of input scintillation data
            CELER_ASSERT(inp);
            // Material-only data
            MaterialScintillationSpectrum mat_spec;
            mat_spec.yield = inp.material.yield;
            CELER_VALIDATE(mat_spec.yield > 0,
                           << "invalid yield=" << mat_spec.yield
                           << " for scintillation (should be positive)");
            auto comps = this->copy_components(inp.material.components);
            mat_spec.components
                = build_compoments.insert_back(comps.begin(), comps.end());
            build_materials.push_back(std::move(mat_spec));
        }
        CELER_ENSURE(build_materials.size() == num_mat);
    }
    else
    {
        // Store particle- and material-dependent scintillation data
        // Index should loop over particles first, then materials. This
        // ordering should improve memory usage, since we group the material
        // list for each particle, and each particle (i.e. stream) traverses
        // multiple materials
        DedupeCollectionBuilder build_reals(&host_data.reals);
        CollectionBuilder build_particles(&host_data.particles);

        for (auto pid : range(num_part))
        {
            auto const pdg = particle_params->id_to_pdg(ParticleId(pid)).get();
            for (auto matid : range(num_mat))
            {
                // Particle- and material-dependend scintillation spectrum
                auto iter = input.data[matid].particles.find(pdg);
                CELER_ASSERT(iter != input.data[matid].particles.end()
                             && pdg == iter->first);

                auto const& ipss = iter->second;
                ParticleScintillationSpectrum part_spec;
                part_spec.yield_vector.grid = build_reals.insert_back(
                    ipss.yield_vector.x.begin(), ipss.yield_vector.x.end());
                part_spec.yield_vector.value = build_reals.insert_back(
                    ipss.yield_vector.y.begin(), ipss.yield_vector.y.end());
                CELER_ASSERT(part_spec.yield_vector);

                auto comps = this->copy_components(ipss.components);
                part_spec.components
                    = build_compoments.insert_back(comps.begin(), comps.end());

                build_particles.push_back(std::move(part_spec));
            }
        }
        CELER_ENSURE(build_particles.size() == num_part * num_mat);
    }

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
    std::vector<ScintillationComponent> comp(input_comp.size());
    real_type norm{0};
    for (auto i : range(comp.size()))
    {
        comp[i].lambda_mean = input_comp[i].lambda_mean;
        comp[i].lambda_sigma = input_comp[i].lambda_sigma;
        comp[i].rise_time = input_comp[i].rise_time;
        comp[i].fall_time = input_comp[i].fall_time;
        norm += input_comp[i].yield;
    }

    // Store normalized yield
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
