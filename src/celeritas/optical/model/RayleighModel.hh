//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/RayleighModel.hh
//---------------------------------------------------------------------------//
#pragma once

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 */
class RayleighModel : public OpticalModel, public ParamsDataInterface<RayleighData>
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstImported = std::shared_ptr<ImportedOpticalMaterials const>;
    //!@}

  public:
    RayleighModel(ActionId id, SPConstImported imported)
        : OpticalModel(id, "rayleigh", "optical rayleigh scattering")
        , imported_(imported)
    {
        CELER_EXPECT(imported_);
    }

    void build_mfp(OpticalModelMfpBuilder& builder) const override final
    {
        builder(imported_->get(builder.optical_material()).rayleigh.mfp);
    }

    void execute(OpticalParams const&, OpticalStateHost&) const override final;
    void execute(OpticalParams const&, OpticalStateDevice&) const override final;

    HostRef const& host_ref() const { return data_.host_ref(); }
    DeviceRef const& device_ref() const { return data_.device_ref(); }

  private:
    CollectionMirror<RayleighData> data_;
    SPConstImported imported_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
}  // namespace celeritas
