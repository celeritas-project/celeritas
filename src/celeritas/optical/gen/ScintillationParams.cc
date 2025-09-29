//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/ScintillationParams.cc
//---------------------------------------------------------------------------//
#include "ScintillationParams.hh"

#include <algorithm>
#include <numeric>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/Types.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/phys/ParticleParams.hh"

#include "detail/MatScintSpecInserter.hh"

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
    CELER_EXPECT(!data.optical_materials.empty());

    if (!std::any_of(data.optical_materials.begin(),
                     data.optical_materials.end(),
                     [](auto const& iter) {
                         return static_cast<bool>(iter.scintillation);
                     }))
    {
        // No scintillation data present
        CELER_LOG(info) << "Skipping scintillation process: no scintillating "
                           "materials are present";
        return nullptr;
    }

    size_type const num_optmats = data.optical_materials.size();

    Input input;
    if (data.optical_params.scintillation_by_particle)
    {
        // Create a mapping of \c ParticleId to \c ScintParticleId
        input.pid_to_scintpid.resize(data.particles.size());
        ScintParticleId spid{0};
        for (auto const& opt_mat : data.optical_materials)
        {
            auto const& pdg_to_spec = opt_mat.scintillation.particles;
            for (auto const& [pdg, spec] : pdg_to_spec)
            {
                if (auto pid = particle_params->find(PDGNumber{pdg}))
                {
                    input.pid_to_scintpid[pid.get()] = spid++;
                }
            }
        }
        // Resize particle- and material-dependent spectra
        input.particles.resize(spid.get() * num_optmats);
    }
    else
    {
        // Resize material-only spectra
        input.materials.resize(num_optmats);
    }

    input.resolution_scale.resize(num_optmats);
    for (auto opt_idx : range(num_optmats))
    {
        ImportOpticalMaterial const& opt_mat = data.optical_materials[opt_idx];
        input.resolution_scale[opt_idx]
            = opt_mat.scintillation.resolution_scale;

        if (!data.optical_params.scintillation_by_particle)
        {
            // Copy the material spectrum
            input.materials[opt_idx] = opt_mat.scintillation.material;
        }
        else
        {
            // Particle and material spectrum
            auto const& pgd_to_spec = opt_mat.scintillation.particles;

            for (auto const& [pdg, spec] : pgd_to_spec)
            {
                if (auto pid = particle_params->find(PDGNumber{pdg}))
                {
                    auto spid = input.pid_to_scintpid[pid.get()];
                    CELER_ASSERT(spid);
                    input.particles[num_optmats * spid.get() + opt_idx] = spec;
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
}  // namespace celeritas
