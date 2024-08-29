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
#include "celeritas/Types.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "detail/MatScintSpecInserter.hh"

namespace celeritas
{
namespace optical
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
        ScintillationParticleId scintpid{0};
        for (auto const& [matid, iom] : data.optical)
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

    size_type optmatidx{0};
    for (auto const& [matid, iom] : data.optical)
    {
        input.resolution_scale[optmatidx] = iom.scintillation.resolution_scale;

        if (!data.optical_params.scintillation_by_particle)
        {
            // Material spectrum
            auto const& iomsm = iom.scintillation.material;
            ImportMaterialScintSpectrum mat_spec;
            mat_spec.yield_per_energy = iomsm.yield_per_energy;
            mat_spec.components = iomsm.components;
            input.materials[optmatidx] = std::move(mat_spec);
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
                    input.particles[num_optmats * scintpid.get() + optmatidx]
                        = std::move(part_spec);
                }
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
    CELER_EXPECT(!input.resolution_scale.empty());
    CELER_VALIDATE(input.particles.empty() != input.materials.empty(),
                   << "conflicting scintillation input: material and particle "
                      "spectra are mutually exclusive");
    CELER_VALIDATE(input.particles.empty(),
                   << "per-particle scintillation spectra are not yet "
                      "implemented");
    CELER_VALIDATE(input.materials.size() == input.resolution_scale.size(),
                   << "material and resolution scales do not match");

    HostVal<ScintillationData> host_data;

    // Store resolution scale
    for (auto const& val : input.resolution_scale)
    {
        CELER_VALIDATE(val >= 0,
                       << "invalid resolution_scale=" << val
                       << " for scintillation (should be nonnegative)");
    }
    CollectionBuilder(&host_data.resolution_scale)
        .insert_back(input.resolution_scale.begin(),
                     input.resolution_scale.end());

    // Store material scintillation data
    detail::MatScintSpecInserter insert_mat{&host_data};
    for (auto const& mat : input.materials)
    {
        insert_mat(mat);
    }
    CELER_ASSERT(host_data.materials.size()
                 == host_data.resolution_scale.size());

    // Copy to device
    mirror_ = CollectionMirror<ScintillationData>{std::move(host_data)};
    CELER_ENSURE(mirror_);
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
