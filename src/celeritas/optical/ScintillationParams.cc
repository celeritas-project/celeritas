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
#include "celeritas/grid/GenericGridBuilder.hh"
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

    auto const& num_optmats = data.optical.size();

    Input input;
    input.resolution_scale.resize(num_optmats);
    if (data.optical_params.scintillation_by_particle)
    {
        // Collect ScintillationParticleIds
        input.pid_to_scintpid.resize(data.particles.size());
        ScintillationParticleId scintpid;
        for (auto const& [matid, iom] : data.optical)
        {
            auto const& iomsp = iom.scintillation.particles;
            for (auto const& [pdg, ipss] : iomsp)
            {
                auto const pid = particle_params->find(PDGNumber{pdg});
                // Add new ScintillationParticleId
                input.pid_to_scintpid[pid.get()] = scintpid++;
            }
        }
        // Resize particle- and material-dependent spectra
        input.particles.resize(input.pid_to_scintpid.size() * num_optmats);
    }
    else
    {
        // Resize material-only spectra
        input.materials.resize(num_optmats);
    }

    size_type optmatidx{0};
    size_type scintpidx{0};
    for (auto const& [matid, iom] : data.optical)
    {
        input.resolution_scale[optmatidx] = iom.scintillation.resolution_scale;

        if (!data.optical_params.scintillation_by_particle)
        {
            // Material spectrum
            auto const& iomsm = iom.scintillation.material;
            ImportMaterialScintSpectrum mat_spec;
            mat_spec.yield = iomsm.yield;
            mat_spec.components = iomsm.components;
            input.materials[optmatidx] = std::move(mat_spec);
        }
        else
        {
            // Particle and material spectrum
            auto const& iomsp = iom.scintillation.particles;

            for (auto const& [pdg, ipss] : iomsp)
            {
                ImportParticleScintSpectrum part_spec;
                part_spec.yield_vector = ipss.yield_vector;
                part_spec.components = ipss.components;
                input.particles[num_optmats * scintpidx + optmatidx]
                    = std::move(part_spec);
                scintpidx++;
            }
        }
        optmatidx++;
    }

    return std::make_shared<ScintillationParams>(std::move(input));
}

//---------------------------------------------------------------------------//
/*!
 * Construct with scintillation input data.
 */
ScintillationParams::ScintillationParams(Input const& input)
{
    CELER_EXPECT(input);
    HostVal<ScintillationData> host_data;
    CollectionBuilder build_resolutionscale(&host_data.resolution_scale);
    CollectionBuilder build_components(&host_data.components);

    // Store resolution scale
    for (auto const& val : input.resolution_scale)
    {
        CELER_EXPECT(!input.resolution_scale.empty());
        CELER_VALIDATE(val >= 0,
                       << "invalid resolution_scale=" << val
                       << " for scintillation (should be nonnegative)");
        build_resolutionscale.push_back(val);
    }
    CELER_ENSURE(!host_data.resolution_scale.empty());

    if (input.particles.empty())
    {
        // Store material scintillation data
        CELER_EXPECT(!input.materials.empty());
        CollectionBuilder build_materials(&host_data.materials);

        for (auto const& mat : input.materials)
        {
            // Check validity of input scintillation data
            CELER_ASSERT(mat);
            // Material-only data
            MaterialScintillationSpectrum mat_spec;
            CELER_VALIDATE(mat.yield > 0,
                           << "invalid yield=" << mat.yield
                           << " for scintillation (should be positive)");
            mat_spec.yield = mat.yield;
            auto comps = this->build_components(mat.components);
            mat_spec.components
                = build_components.insert_back(comps.begin(), comps.end());
            build_materials.push_back(std::move(mat_spec));
        }
        CELER_VALIDATE(input.materials.size() == input.resolution_scale.size(),
                       << "material and resolution scales do not match.");
        CELER_ENSURE(host_data.materials.size()
                     == host_data.resolution_scale.size());
    }
    else
    {
        // Store particle data
        CELER_EXPECT(!input.pid_to_scintpid.empty());
        CollectionBuilder build_scintpid(&host_data.pid_to_scintpid);

        // Store particle ids
        for (auto const id : input.pid_to_scintpid)
        {
            build_scintpid.push_back(id);
            if (id)
            {
                host_data.num_scint_particles++;
            }
        }
        CELER_ENSURE(host_data.num_scint_particles > 0);

        // Store particle spectra
        CELER_EXPECT(!input.particles.empty());
        GenericGridBuilder build_grid(&host_data.reals);
        CollectionBuilder build_particles(&host_data.particles);

        for (auto spec : input.particles)
        {
            ParticleScintillationSpectrum part_spec;
            part_spec.yield_vector = build_grid(spec.yield_vector);
            auto comps = this->build_components(spec.components);
            part_spec.components
                = build_components.insert_back(comps.begin(), comps.end());
            build_particles.push_back(std::move(part_spec));
        }
        CELER_ENSURE(host_data.particles.size()
                     == host_data.num_scint_particles
                            * host_data.resolution_scale.size());
    }

    // Copy to device
    mirror_ = CollectionMirror<ScintillationData>{std::move(host_data)};
    CELER_ENSURE(mirror_);
}

//---------------------------------------------------------------------------//
/*!
 * Return a \c ScintillationComponent from a \c ImportScintComponent .
 */
std::vector<ScintillationComponent> ScintillationParams::build_components(
    std::vector<ImportScintComponent> const& input_comp)
{
    std::vector<ScintillationComponent> comp(input_comp.size());
    real_type norm{0};
    for (auto i : range(comp.size()))
    {
        CELER_VALIDATE(input_comp[i].lambda_mean > 0,
                       << "invalid lambda_mean=" << input_comp[i].lambda_mean
                       << " for scintillation component (should be positive)");
        CELER_VALIDATE(input_comp[i].lambda_sigma > 0,
                       << "invalid lambda_sigma=" << input_comp[i].lambda_sigma
                       << " for scintillation component " << i
                       << " (should be positive)");
        CELER_VALIDATE(input_comp[i].rise_time >= 0,
                       << "invalid rise_time=" << input_comp[i].rise_time
                       << " for scintillation component " << i
                       << " (should be "
                          "nonnegative)");
        CELER_VALIDATE(input_comp[i].fall_time > 0,
                       << "invalid fall_time=" << input_comp[i].fall_time
                       << " for scintillation component " << i
                       << " (should be positive)");
        comp[i].lambda_mean = input_comp[i].lambda_mean;
        comp[i].lambda_sigma = input_comp[i].lambda_sigma;
        comp[i].rise_time = input_comp[i].rise_time;
        comp[i].fall_time = input_comp[i].fall_time;
        norm += input_comp[i].yield;
    }

    // Store normalized yield
    for (auto i : range(comp.size()))
    {
        CELER_VALIDATE(input_comp[i].yield > 0,
                       << "invalid yield=" << input_comp[i].yield
                       << " for scintillation component " << i);
        comp[i].yield_frac = input_comp[i].yield / norm;
    }
    return comp;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
