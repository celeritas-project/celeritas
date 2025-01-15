//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/AbsorptionModel.cc
//---------------------------------------------------------------------------//
#include "AbsorptionModel.hh"

#include "corecel/Assert.hh"
#include "celeritas/io/ImportOpticalMaterial.hh"

#include "../MfpBuilder.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
/*!
 * Create a model builder for the optical absorption model.
 */
ModelBuilder AbsorptionModel::make_builder(SPConstImported imported)
{
    CELER_EXPECT(imported);
    return [i = std::move(imported)](ActionId id) {
        return std::make_shared<AbsorptionModel>(id, i);
    };
}

//---------------------------------------------------------------------------//
/*!
 * Construct the model from imported data.
 */
AbsorptionModel::AbsorptionModel(ActionId id, SPConstImported imported)
    : Model(id, "absorption", "interact by optical absorption")
    , imported_(ImportModelClass::absorption, imported)
{
}

//---------------------------------------------------------------------------//
/*!
 * Build the mean free paths for the model.
 */
void AbsorptionModel::build_mfps(OpticalMaterialId mat, MfpBuilder& build) const
{
    CELER_EXPECT(mat < imported_.num_materials());
    build(imported_.mfp(mat));
}

//---------------------------------------------------------------------------//
/*!
 * Execute the model on the host.
 */
void AbsorptionModel::step(CoreParams const&, CoreStateHost&) const
{
    CELER_NOT_IMPLEMENTED("optical core physics");
}

//---------------------------------------------------------------------------//
/*!
 * Execute the model on the device.
 */
#if !CELER_USE_DEVICE
void AbsorptionModel::step(CoreParams const&, CoreStateDevice&) const
{
    CELER_NOT_CONFIGURED("CUDA OR HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
