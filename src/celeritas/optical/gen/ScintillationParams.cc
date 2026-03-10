//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/ScintillationParams.cc
//---------------------------------------------------------------------------//
#include "ScintillationParams.hh"

#include <algorithm>

#include "corecel/cont/Range.hh"
#include "corecel/data/CollectionBuilder.hh"
#include "celeritas/Types.hh"
#include "celeritas/optical/MaterialParams.hh"

#include "detail/ScintSpectrumInserter.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct with optical materials and scintillation process data.
 */
ScintillationParams::ScintillationParams(
    optical::MaterialParams const& optical_mat,
    inp::ScintillationProcess const& process)
{
    CELER_EXPECT(!process.materials.empty());
    CELER_EXPECT(optical_mat.num_materials() > 0);

    size_type const num_optmats = optical_mat.num_materials();

    HostVal<ScintillationData> host_data;

    // Initialize resolution scale for all materials (default to 1.0)
    std::vector<real_type> resolution_scale(num_optmats, 1.0);

    // Store material scintillation data
    detail::ScintSpectrumInserter insert_mat{&host_data};
    for (auto opt_id :
         range(OptMatId{static_cast<OptMatId::size_type>(num_optmats)}))
    {
        auto iter = process.materials.find(opt_id);
        if (iter != process.materials.end())
        {
            // Material has scintillation data
            auto const& scint_mat = iter->second;
            resolution_scale[opt_id.get()] = scint_mat.resolution_scale;
            insert_mat(scint_mat);
        }
        else
        {
            // Material has no scintillation: insert empty material
            insert_mat();
        }
    }

    // Store resolution scale
    CollectionBuilder(&host_data.resolution_scale)
        .insert_back(resolution_scale.begin(), resolution_scale.end());

    CELER_ASSERT(host_data.spectra.size() == host_data.resolution_scale.size());

    // Copy to device
    mirror_ = ParamsDataStore<ScintillationData>{std::move(host_data)};
    CELER_ENSURE(mirror_);
}

//---------------------------------------------------------------------------//
/*!
 * Whether any celeritas-only features are present.
 */
bool ScintillationParams::is_geant_compatible() const
{
    auto const& scint_records
        = this->host_ref().scint_records[AllItems<ScintDistributionRecord>{}];
    return std::all_of(scint_records.begin(),
                       scint_records.end(),
                       [](ScintDistributionRecord const& sr) {
                           return !sr.is_normal_distribution();
                       });
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
