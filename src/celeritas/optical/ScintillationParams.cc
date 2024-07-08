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
    CELER_EXPECT(!data.opt_materials.empty());

    if (!std::any_of(data.opt_materials.begin(),
                     data.opt_materials.end(),
                     [](auto const& opt_mat) {
                         return static_cast<bool>(opt_mat.scintillation);
                     }))
    {
        // No scintillation data present
        return nullptr;
    }

    auto const& num_optmats = data.opt_materials.size();

    Input input;
    input.resolution_scale.resize(num_optmats);
    if (data.optical_params.scintillation_by_particle)
    {
        // Collect ScintillationParticleIds
        input.pid_to_scintpid.resize(data.particles.size());
        ScintillationParticleId scintpid{0};
        for (auto const& iom : data.opt_materials)
        {
            auto const& iomsp = iom.scintillation.particles;
            for (auto const& [pdg, ipss] : iomsp)
            {
                if (auto const pid = particle_params->find(PDGNumber{pdg}))
                {
                    // Add new ScintillationParticleId
                    input.pid_to_scintpid[pid.get()] = scintpid++;
                }
            }
        }
        // Resize particle- and material-dependent spectra
        auto const num_scint_particles = scintpid.get();
        input.particles.resize(num_scint_particles * num_optmats);
    }
    else
    {
        // Resize material-only spectra
        input.materials.resize(num_optmats);
    }

    for (auto optmatidx : range(OpticalMaterialId{num_optmats}))
    {
        auto const& iom = data.opt_materials[optmatidx.get()];
        input.resolution_scale[optmatidx.get()]
            = iom.scintillation.resolution_scale;

        if (!data.optical_params.scintillation_by_particle)
        {
            // Material spectrum
            auto const& iomsm = iom.scintillation.material;
            ImportMaterialScintSpectrum mat_spec;
            mat_spec.yield_per_energy = iomsm.yield_per_energy;
            mat_spec.components = iomsm.components;
            input.materials[optmatidx.get()] = std::move(mat_spec);
        }
        else
        {
            // Particle and material spectrum
            auto const& iomsp = iom.scintillation.particles;

            for (auto const& [pdg, ipss] : iomsp)
            {
                if (auto const pid = particle_params->find(PDGNumber{pdg}))
                {
                    auto scintpid = input.pid_to_scintpid[pid.get()];
                    CELER_ASSERT(scintpid);
                    ImportParticleScintSpectrum part_spec;
                    part_spec.yield_vector = ipss.yield_vector;
                    part_spec.components = ipss.components;
                    input.particles[num_optmats * scintpid.get()
                                    + optmatidx.get()]
                        = std::move(part_spec);
                }
            }
        }
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
    CELER_VALIDATE(input.particles.empty() != input.materials.empty(),
                   << "invalid input data. Material spectra and particle "
                      "spectra are mutually exclusive. Please store either "
                      "material or particle spectra, but not both.");

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
        CollectionBuilder build_materials(&host_data.materials);

        for (auto const& mat : input.materials)
        {
            // Check validity of input scintillation data
            CELER_ASSERT(mat);
            // Material-only data
            MaterialScintillationSpectrum mat_spec;
            CELER_VALIDATE(mat.yield_per_energy > 0,
                           << "invalid yield=" << mat.yield_per_energy
                           << " for scintillation (should be positive)");
            mat_spec.yield_per_energy = mat.yield_per_energy;
            auto comps = this->build_components(mat.components);
            mat_spec.components
                = build_components.insert_back(comps.begin(), comps.end());
            build_materials.push_back(std::move(mat_spec));
        }
        CELER_VALIDATE(input.materials.size() == input.resolution_scale.size(),
                       << "material and resolution scales do not match");
        CELER_ENSURE(host_data.materials.size()
                     == host_data.resolution_scale.size());
    }
    else
    {
        // Store particle data
        CELER_VALIDATE(!input.pid_to_scintpid.empty(),
                       << "missing particle ID to scintillation particle ID "
                          "mapping");
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
            CELER_VALIDATE(spec.yield_vector,
                           << "particle yield vector is not assigned "
                              "correctly");
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
        norm += input_comp[i].yield_per_energy;
    }

    // Store normalized yield
    for (auto i : range(comp.size()))
    {
        CELER_VALIDATE(input_comp[i].yield_per_energy > 0,
                       << "invalid yield=" << input_comp[i].yield_per_energy
                       << " for scintillation component " << i);
        comp[i].yield_frac = input_comp[i].yield_per_energy / norm;
    }
    return comp;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
