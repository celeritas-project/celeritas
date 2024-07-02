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
    using SPConstImported = std::shared_ptr<ImportedOpticalMaterials const>;
    //!@}

  public:
    AbsorptionModel(ActionId id, SPConstImported imported)
        : OpticalModel(id, "absorption", "volumetric optical absorption")
        , imported_(imported)
    {}

    void build_mfp(OpticalModelMfpBuilder& build) const override final
    {
        builder(imported_->get(builder.optical_material()).absorption.mfp);
    }

    void execute(OpticalParams const&, OpticalStateHost&) const override final;
    void execute(OpticalParams const&, OpticalStateDevice&) const override final;

  private:
    SPConstImported imported_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 */

//---------------------------------------------------------------------------//
}  // namespace celeritas
