//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/OpticalRayleighModel.cc
//---------------------------------------------------------------------------//
#include "OpticalRayleighModel.hh"

#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/optical/OpticalMfpBuilder.hh"
#include "celeritas/optical/OpticalModelBuilder.hh"
#include "celeritas/optical/OpticalPhysicsParams.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Construct the model from imported data.
 */
OpticalRayleighModel::OpticalRayleighModel(
    ActionId id, std::vector<ImportOpticalMaterial> const& imported)
    : OpticalModel(id, "rayleigh", "optical rayleigh scattering"), imported_()
{
    imported_.reserve(imported.size());

    HostVal<OpticalRayleighData> data;
    CollectionBuilder<real_type, MemSpace::host, OpticalMaterialId>
        build_scale_factor{&data.scale_factor};
    CollectionBuilder<real_type, MemSpace::host, OpticalMaterialId>
        build_compressibility{&data.compressibility};

    for (auto const& opt_mat : imported)
    {
        auto const& import_rayleigh = opt_mat.rayleigh;

        imported_.push_back(import_rayleigh);
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
    CELER_EXPECT(builder.optical_material().get() < imported_.size());
    builder(imported_[builder.optical_material().get()].mfp);
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
