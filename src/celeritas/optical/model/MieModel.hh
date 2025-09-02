//-----------------------------------------------*-C++-*----------------------------------//
// Copyright ...
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/model/MieModel.hh
//! \brief Optical Mie scattering process (Henyey-Greenstein) model
////---------------------------------------------------------------------------///*
// #pragma once
//
// #include "celeritas/optical/ImportedModelAdapter.hh"
// #include "celeritas/optical/Model.hh"
// #include "celeritas/optical/Types.hh"
//
// namespace celeritas
//{
// class MaterialParams;
// struct ImportOpticalMie;
//
// namespace optical
//{
//     class ImportedMaterials;
//     class MaterialParams;
////---------------------------------------------------------------------------//
///*!
// * Mie scattering model for optical photons.
// *
// * This is a placeholder that will eventually implement Henyey–Greenstein
// * scattering (like Geant4’s G4OpMieHG). For now, it inserts an empty MFP
// grid
// * and logs a debug message.
// */
// class MieModel final : public Model
//{
//  public:
//    //!@{
//    //! \name Type aliases
//    using SPConstImported = std::shared_ptr<ImportedModels const>;
//    using SPConstImportedMaterials = std::shared_ptr<ImportedMaterials
//    const>; using SPConstMaterials = std::shared_ptr<MaterialParams const>;
//    using SPConstCoreMaterials
//        = std::shared_ptr<::celeritas::MaterialParams const>;
//    //!@}
// struct Input
//    {
//        SPConstMaterials materials;
//        SPConstCoreMaterials core_materials;
//        SPConstImportedMaterials imported_materials;
//    };
//    //!@}
//
//  public:
//   static ModelBuilder make_builder(SPConstImported, Input);
//    // Construct from imported data and optional input
//    MieModel(ActionId id, SPConstImported imported, Input input);
//
//    // Build mean free path grid for this model
//    void build_mfps(OptMatId mat, MfpBuilder& build) const final;
//
//    // Step function: host
//    void step(CoreParams const& params, CoreStateHost& state) const final;
//
//    // Step function: device
//    void step(CoreParams const& params, CoreStateDevice& state) const final;
//
//  private:
//    ImportedModelAdapter imported_;
//    Input input_;
//};
//
////---------------------------------------------------------------------------//
//}  // namespace optical
//}  // namespace celeritas
//---------------------------------*- C++
//-*----------------------------------//
//---------------------------------*- C++
//-*----------------------------------//
#pragma once

#include "celeritas/io/ImportOpticalMaterial.hh"
#include "celeritas/optical/ImportedModelAdapter.hh"
#include "celeritas/optical/Model.hh"

namespace celeritas
{
class MaterialParams;
struct ImportMie;
// struct ImportOpticalRayleigh;

namespace optical
{

class ImportedMaterials;
class MaterialParams;

//---------------------------------------------------------------------------//
class MieModel final : public Model
{
  public:
    using SPConstImported = std::shared_ptr<ImportedModels const>;
    using SPConstImportedMaterials = std::shared_ptr<ImportedMaterials const>;
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;
    using SPConstCoreMaterials
        = std::shared_ptr<::celeritas::MaterialParams const>;
    struct Input
    {
        SPConstMaterials materials;
        SPConstCoreMaterials core_materials;
        SPConstImportedMaterials imported_materials;
        explicit operator bool() const
        {
            return materials && core_materials && imported_materials;
        }
    };

    static ModelBuilder make_builder(SPConstImported imported, Input input);

    MieModel(ActionId id, SPConstImported imported, Input input);

    void build_mfps(OptMatId mat, MfpBuilder& build) const final;
    void step(CoreParams const&, CoreStateHost&) const final;
    void step(CoreParams const&, CoreStateDevice&) const final;

  private:
    ImportedModelAdapter imported_;
    Input input_;
};

}  // namespace optical
}  // namespace celeritas
