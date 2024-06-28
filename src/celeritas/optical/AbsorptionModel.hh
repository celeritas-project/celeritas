//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/AbsorptionModel.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 */
class AbsorptionModel : public OpticalModel
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstImported = std::shared_ptr<ImportedOpticalModels const>;
    //!@}

  public:
    AbsorptionModel(ActionId id, SPConstImported imported);

    StepLimitBuilder step_limits(OpticalMaterialId opt_mat) const override final;

    void execute(OpticalParams const&, OpticalStateHost&) const override final;
    void execute(OpticalParams const&, OpticalStateDevice&) const override final;

  private:
    ImportedOpticalModelAdapter imported_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 */

//---------------------------------------------------------------------------//
}  // namespace celeritas
