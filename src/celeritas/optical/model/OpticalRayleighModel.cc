//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/OpticalRayleighModel.cc
//---------------------------------------------------------------------------//
#include "OpticalRayleighModel.hh"

#include "celeritas/optical/ImportedOpticalMaterials.hh"
#include "celeritas/optical/OpticalMfpBuilder.hh"
#include "celeritas/optical/OpticalModelBuilder.hh"
#include "celeritas/optical/OpticalPhysicsParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct the model from imported data.
 */
OpticalRayleighModel::OpticalRayleighModel(ActionId id,
                                           SPConstImported imported)
    : OpticalModel(id, "rayleigh", "optical rayleigh scattering")
    , imported_(imported)
{
    CELER_EXPECT(imported_);

    HostVal<OpticalRayleighData> data;
    CollectionBuilder<real_type, MemSpace::host, OpticalMaterialId>
        build_scale_factor{&data.scale_factor};
    CollectionBuilder<real_type, MemSpace::host, OpticalMaterialId>
        build_compressibility{&data.compressibility};

    for (auto opt_mat_id : range(OpticalMaterialId{imported_->size()}))
    {
        auto const& import_rayleigh = imported_->get(opt_mat_id).rayleigh;

        build_scale_factor.push_back(import_rayleigh.scale_factor);
        build_compressibility.push_back(import_rayleigh.compressibility);
    }

    data_ = CollectionMirror<OpticalRayleighData>{std::move(data)};
}

//---------------------------------------------------------------------------//
/*!
 * Build mean free paths for the model.
 */
void OpticalRayleighModel::build_mfp(OpticalModelMfpBuilder& builder) const
{
    builder(imported_->get(builder.optical_material()).rayleigh.mfp);
}

//---------------------------------------------------------------------------//
/*!
 * Execute model on host.
 */
void OpticalRayleighModel::execute(OpticalParams const&, OpticalStateHost&) const
{
}

//---------------------------------------------------------------------------//
/*!
 * Execute model on device.
 */
#if !CELER_USE_DEVICE
void OpticalRayleighModel::execute(OpticalParams const&,
                                   OpticalStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace celeritas
